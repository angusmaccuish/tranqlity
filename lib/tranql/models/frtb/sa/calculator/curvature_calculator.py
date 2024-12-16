from typing import Dict, Union

import numpy as np
import pandas as pd

from .sbm_calculator import SBMCalculator


class CurvatureCalculator(SBMCalculator):
    """
    Base class for all Curvature Calculators
    """

    def get_risk_weights(self, labels: pd.Series) -> pd.Series:
        return labels.map(self.risk_weights)

    def get_weighted_sensis(self, sensis: pd.DataFrame) -> Dict[str, pd.Series]:
        flatten = self.params.get('sa.CurvatureFlag', 'Y').upper() == 'N'

        if flatten:
            condition = (sensis['CurvatureFlag'] != 'Y') & (sensis['CurvatureFlag'] != 'y')
            sensis.loc[condition, 'SHIFTUPPNL'] = 0.0
            sensis.loc[condition, 'SHIFTDOWNPNL'] = 0.0
            sensis.loc[condition, 'CURVDELTA'] = 0.0

        ws = sensis['CURVDELTA'] * sensis['RiskWeightValue']

        return {
            'cvru': -(sensis['SHIFTUPPNL'] - ws),
            'cvrd': -(sensis['SHIFTDOWNPNL'] + ws),
            'ws': ws,
        }

    def net_weighted_sensis(self, data: pd.DataFrame) -> pd.DataFrame:
        agg_cols = ['FRTBBucket', 'GroupID', 'Underlying']
        return data.groupby(by=agg_cols, sort=False)[['cvru', 'cvrd', 'ws']].sum()

    def calc_intra_bucket(self, ws: pd.DataFrame, scenario: str) -> pd.DataFrame:
        ws['cvru_pos'] = ws['cvru'].clip(lower=0.0)
        ws['cvrd_pos'] = ws['cvrd'].clip(lower=0.0)
        ws['cvru_neg'] = ws['cvru'].clip(upper=0.0)
        ws['cvrd_neg'] = ws['cvrd'].clip(upper=0.0)
        ws['cvru_pos2'] = np.square(ws['cvru_pos'])
        ws['cvrd_pos2'] = np.square(ws['cvrd_pos'])

        kb = ws.groupby(level=['FRTBBucket', 'GroupID'], sort=False).sum()

        # Look up (scalar) rho correlation per bucket (zero for 'other' bucket, easier to do this rather than filter)
        buckets = ws.index.get_level_values('FRTBBucket').unique()
        rhos = {bucket: self._rho_correlation(scenario, bucket) for bucket in buckets if bucket != self.other_bucket}
        rhos[self.other_bucket] = 0.0
        rho = kb.index.get_level_values('FRTBBucket').map(rhos)

        kb['kbu'] = kb['cvru_pos2']
        kb['xbu'] = rho * (np.square(kb['cvru']) - np.square(kb['cvru_neg']) - kb['cvru_pos2'])

        kb['kbd'] = kb['cvrd_pos2']
        kb['xbd'] = rho * (np.square(kb['cvrd']) - np.square(kb['cvrd_neg']) - kb['cvrd_pos2'])

        # Other bucket
        if self.other_bucket is not None and self.other_bucket in kb.index:
            kb.loc[self.other_bucket, 'kbu'] = np.square(kb['cvru_pos'])
            kb.loc[self.other_bucket, 'xbu'] = 0.0
            kb.loc[self.other_bucket, 'kbd'] = np.square(kb['cvrd_pos'])
            kb.loc[self.other_bucket, 'xbd'] = 0.0

        kb['ssbu'] = kb['kbu']
        kb['ssbd'] = kb['kbd']
        kb['sbu'] = kb['cvru'].clip(lower=0.0)
        kb['sbd'] = kb['cvrd'].clip(lower=0.0)
        kb['kbu'] = kb['kbu'] + kb['xbu']
        kb['kbd'] = kb['kbd'] + kb['xbd']
        kb['kbu'] = np.sqrt(np.maximum(kb['kbu'], 0.0))
        kb['kbd'] = np.sqrt(np.maximum(kb['kbd'], 0.0))
        kb['kb'] = np.where(kb['kbu'] > kb['kbd'], kb['kbu'], kb['kbd'])

        return kb

    def calculate_sb(self, kb: pd.DataFrame):
        kb['sb'] = np.where(kb['kbu'] > kb['kbd'], kb['cvru'], kb['cvrd'])
        kb['sb'] = np.where(kb['kbu'] == kb['kbd'], np.maximum(kb['cvru'], kb['cvrd']), kb['sb'])

    def calc_inter_bucket(self, kb: pd.DataFrame, scenario: str) -> pd.DataFrame:
        buckets = kb.index.get_level_values('FRTBBucket').unique()
        group_ids = kb.index.get_level_values('GroupID').unique()

        gamma_correlations = self.gamma_correlations()
        gamma, default_gamma = gamma_correlations[scenario]

        if gamma is not None:
            gamma = gamma.reindex(index=buckets, columns=buckets, fill_value=default_gamma).to_numpy()
            np.fill_diagonal(gamma, 0.0)

        def correlate(sb: pd.Series):
            if gamma is None:
                # Constant off-diagonal correlation, reduces to square of sums minus sum of squares formula
                df_ = pd.DataFrame({'sb': sb, 'sb2': np.square(sb)}).groupby(level='GroupID', sort=False).sum()
                return default_gamma * (np.square(df_['sb']) - df_['sb2'])
            else:
                # Use explicit gamma matrix
                # Construct a matrix of the inputs to the correlation
                # Explicitly align rows with gamma buckets, columns with groups as unstack sorts index by default,
                # current version of pandas does not support the new sort=False option
                matrix = sb.unstack('GroupID', fill_value=0.0).loc[buckets, group_ids]
                return np.einsum('ji,jk,ki->i', matrix, gamma, matrix)

        # Include all the core measures for now and filter later (we could just construct the result df more carefully)
        df = kb.groupby(level='GroupID', sort=False)[['cvru', 'cvrd', 'ws']].sum()
        ss = np.square(kb['kb']).groupby(level='GroupID', sort=False).sum().to_numpy()

        # Calculate k^2, deducting the psi terms
        k2 = ss + correlate(kb['sb']) - correlate(kb['sb'].clip(upper=0.0))

        x = k2 - ss
        k = np.sqrt(np.maximum(k2, 0.0))

        return df.assign(k=k, ss=ss, x=x)

    def _rho_correlation(self, scenario: str, bucket: Union[str, None]) -> np.float64:
        rhos = self.rho_correlations(bucket=bucket, rows='Underlying1Value')
        rho = rhos[scenario]
        return rho.loc['Different']


class SimpleCurvatureCalculator(CurvatureCalculator):
    """
    Base class for Risk Classes which have only one underlying per bucket, ie no rho cross-correlation
    """
    def calc_intra_bucket(self, ws: pd.DataFrame, scenario: str) -> pd.DataFrame:
        kb = ws.groupby(level=['FRTBBucket', 'GroupID'], sort=False)[['cvru', 'cvrd', 'ws']].sum()

        kb['kbu'] = kb['sbu'] = kb['cvru'].clip(lower=0.0)
        kb['kbd'] = kb['sbd'] = kb['cvrd'].clip(lower=0.0)
        kb['kb'] = np.where(kb['kbu'] > kb['kbd'], kb['kbu'], kb['kbd'])
        kb['ssbu'] = np.square(kb['kbu'])
        kb['ssbd'] = np.square(kb['kbd'])
        kb['xbu'] = kb['xbd'] = 0.0

        return kb
