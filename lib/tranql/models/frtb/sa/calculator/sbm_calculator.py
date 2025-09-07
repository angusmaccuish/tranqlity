from abc import abstractmethod, ABC
from functools import cache
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .calculator import Calculator


class SBMCalculator(Calculator):
    """
    Base class for all Delta, Vega and Curvature Calculators
    """

    include_all: bool = False
    other_bucket: str = None
    other_bucket_property: str = None

    def __init__(self, *args, **kwargs):
        super(SBMCalculator, self).__init__(*args, **kwargs)

        if self.correlations is None or self.correlations.empty:
            raise Exception('No correlations found')

        # Convert correlations into more manageable rho and gamma DataFrames
        self.rho, self.gamma = self.__normalize_correlations()

        # Check if we need to include correlation terms in the result
        self.include_all = any(m in self.params['measure'] for m in ('ssb', 'xb'))

        if self.other_bucket_property:
            self.other_bucket = str(self.params[self.other_bucket_property])

    @abstractmethod
    def get_risk_weights(self, labels: pd.Series) -> pd.Series:
        """
        Convert the Series of RWLabels into a corresponding Series of risk weights (which may be arrays)
        :param labels: the RWLabels
        :return: the associated risk weights
        """
        pass

    @abstractmethod
    def get_weighted_sensis(self, sensis: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Apply the risk weights to the risk values
        :param sensis: the DataFrame containing the risk and risk weight columns
        :return: a Dictionary of column name -> weighted risk Series, which will be used to create new columns in sensis
        """
        pass

    @abstractmethod
    def net_weighted_sensis(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate the weighted sensis. At the very least this should be at the ('FRTBBucket', 'GroupID') level, but
        most risk classes/type will require additional aggregation levels (e.g. 'Underlying'). Depends on the axes
        required in the intra-bucket (rho) correlation calculation.
        :param data: the input DataFrame containing the weighted risk
        :return: indexed DataFrame containing the net (aggregated) weighted risk
        """
        pass

    def calculate_levels(self, sensis):
        results = {}
        measures = set(self.params['measure'])
        scenario_measures = set(self.params['scenario_measure'])

        # Ensure FRTBBucket is a string
        sensis['FRTBBucket'] = sensis['FRTBBucket'].astype(str)

        # Apply risk weights
        sensis['RiskWeightValue'] = self.get_risk_weights(labels=sensis['RWLabel']).fillna(value=0.0)
        sensis = sensis.assign(**self.get_weighted_sensis(sensis))

        # Get the net weighted sensis, indexed on FRTBBucket and GroupID
        ws = self.net_weighted_sensis(data=sensis)

        # If the client has not requested risk position/charge measures, we may not have scenarios - default to medium
        scenarios = self.params.get('scenario') or ['medium']

        for scenario in scenarios:
            # Do the intra-bucket calculations.
            kb = self.calc_intra_bucket(ws=ws, scenario=scenario)

            # Calculate S_b value. It is not always needed. It depends on the regulation i.e. reg_name.
            self.calculate_sb(kb)

            # Do the inter-bucket calculations.
            k = self.calc_inter_bucket(kb=kb, scenario=scenario)

            for df in k, kb:
                # Drop any intermediate measures we collected along the way
                drop_columns = [col for col in df.columns if col not in measures]
                if drop_columns:
                    df.drop(columns=drop_columns, inplace=True)

                # Rename scenario measures eg k -> k_medium, for example
                renames = {col: f'{col}_{scenario}' for col in df.columns if col in scenario_measures}
                if renames:
                    df.rename(columns=renames, inplace=True)

            results[scenario] = {'total': k, 'bucket': kb}

        return results

    @abstractmethod
    def calc_intra_bucket(self, ws: pd.DataFrame, scenario: str) -> pd.DataFrame:
        """
        Intra-bucket (rho) correlation
        :param ws: indexed DataFrame of weighted risk
        :param scenario: low, medium or high
        :return: DataFrame with bucket-level risk position included (plus additional supporting measures if requested)
        """
        pass

    @abstractmethod
    def calculate_sb(self, kb: pd.DataFrame):
        """
        Adds the sb (S_b in the spec) column to the input DataFrame, different for delta/vega and curvature.
        :param kb: the input DataFrame
        :return: DataFrame with sb column included
        """
        pass

    @abstractmethod
    def calc_inter_bucket(self, kb: pd.DataFrame, scenario: str) -> pd.DataFrame:
        """
        Inter-bucket (gamma) correlation
        :param kb: the input DataFrame (contains intra-bucket calculations)
        :param scenario: low, medium or high
        :return: DataFrame with risk charge added (plus additional columns to help explain the result)
        """
        pass

    def reduce(self, results: Dict):
        # Handle duplicate columns at each level (namely, the non-scenario measures such as ws)
        for level in self.params['level']:
            columns = set()
            for result in results:
                df = results[result][level]
                drop_columns = [col for col in df.columns if col in columns]
                if drop_columns:
                    df.drop(columns=drop_columns, inplace=True)
                columns.update(df.columns)

        # Merge scenario results using concat on GroupID index
        return {
            level: pd.concat([results[result][level] for result in results], axis=1).reset_index()
            for level in self.params['level']
        }

    @cache
    def rho_correlations(self,
                         bucket: Optional[str],
                         rows: str | Iterable,
                         columns: Optional[str | Iterable] = None,
                         fill_value: Optional[float] = None) -> Dict[str, pd.DataFrame]:
        rhos = {}

        if bucket is not None:
            rho = self.rho.loc[bucket] if bucket in self.rho.index else self.rho.loc['***']
        else:
            # Use generic config
            rho = self.rho.loc['***']

        rows = [rows] if isinstance(rows, str) else rows
        columns = [] if columns is None else [columns] if isinstance(columns, str) else columns

        for scenario in 'low', 'medium', 'high':
            result = rho.loc[scenario].set_index(keys=[*rows, *columns])['correlation'].unstack([*columns])

            if fill_value is not None:
                result.fillna(fill_value, inplace=True)

            rhos[scenario] = result

        return rhos

    @cache
    def gamma_correlations(self) -> Dict[str, Tuple]:
        gammas = {}

        for scenario in 'low', 'medium', 'high':
            gamma = self.gamma.loc[scenario]

            default = '***', '***'
            default_gamma = gamma.loc[default, 'correlation'] if default in gamma.index else 0.0
            specific_filter = [(left, right) for left, right in gamma.index if left != right]

            if specific_filter:
                specific_gammas = gamma.loc[specific_filter, 'correlation']
                specific_gammas = specific_gammas.unstack('FRTBBucketY', fill_value=default_gamma)
                gammas[scenario] = specific_gammas, default_gamma
            else:
                gammas[scenario] = None, default_gamma

        return gammas

    def __normalize_correlations(self):
        # Convert correlations into more manageable, normalized rho and gamma DataFrames
        scenarios = ['low', 'medium', 'high']
        correlations = self.correlations.rename(columns={f'CorrelationValue{s.capitalize()}': s for s in scenarios})
        correlations = correlations.astype({'FRTBBucketX': str, 'FRTBBucketY': str})
        id_vars = [
            'CorrelationType', 'FRTBBucketX', 'FRTBBucketY',
            'Underlying1Value', 'Underlying2Value', 'Underlying3Value', 'Underlying4Value'
        ]
        correlations = correlations.loc[:, [*id_vars, *scenarios]]
        correlations = pd.melt(frame=correlations, id_vars=id_vars, var_name='scenario', value_name='correlation')

        is_rho = correlations['CorrelationType'] == 'RHO'
        rho = correlations.loc[is_rho].set_index(['FRTBBucketX', 'scenario'])

        is_gamma = correlations['CorrelationType'] == 'GAMMA'
        gamma = correlations.loc[is_gamma].set_index(['scenario', 'FRTBBucketX', 'FRTBBucketY'])

        return rho, gamma


class OtherBucketOverride(SBMCalculator, ABC):
    """
    Inter-bucket correlation override - split bucket risk position DataFrame into two, one with all the standard
    buckets, and one containing the 'other' bucket only. Apply the usual inter-bucket correlation to the former, and
    then add the latter to this result.
    """

    def calc_inter_bucket(self, kb: pd.DataFrame, scenario: str) -> pd.DataFrame:
        if self.other_bucket not in kb.index:
            # Other bucket not involved, calculation should proceed as normal
            return super(OtherBucketOverride, self).calc_inter_bucket(kb=kb, scenario=scenario)

        # Other bucket risk
        kb_other = kb.loc[self.other_bucket]
        ws_other = kb_other['ws']
        k_other = kb_other['kb']
        ss_other = np.square(k_other)

        buckets = [b for b in kb.index.get_level_values('FRTBBucket').unique() if b != self.other_bucket]

        if buckets:
            # Calculate the inter-bucket correlation for the standard buckets then add the other bucket charge, k
            result = super(OtherBucketOverride, self).calc_inter_bucket(kb=kb.loc[buckets], scenario=scenario)
            group_ids = kb.index.get_level_values('GroupID').unique()
            result = result.reindex(index=group_ids, fill_value=0.0)
            result['ws'] = result['ws'].add(ws_other, fill_value=0.0)
            result['k'] = result['k'].add(k_other, fill_value=0.0)
            return result
        else:
            # Other bucket only, construct result from Series
            return pd.DataFrame.from_dict({'ws': ws_other, 'ss': ss_other, 'k': k_other}).assign(x=0.0, alt_flag=0)
