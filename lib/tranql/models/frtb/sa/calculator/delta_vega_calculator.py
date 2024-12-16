from abc import abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Union

from .sbm_calculator import SBMCalculator


class DeltaVegaCalculator(SBMCalculator):
    """
    Base class for all Delta and Vega Calculators (contains the linearised intra and inter-bucket algorithms)
    """

    has_term_structure: bool
    bucket_dependent_rho_correlation: bool

    def get_risk_weights(self, labels: pd.Series) -> pd.Series:
        if self.has_term_structure:
            # Usually we have the same small set of RW Labels, map unique labels for slight speed-up
            get_risk_weight = self.risk_weights.get
            labels = labels.map(tuple)
            risk_weights = {_labels: np.array(list(map(get_risk_weight, _labels))) for _labels in labels.unique()}
            return labels.map(risk_weights)
        else:
            return labels.map(self.risk_weights)

    def get_weighted_sensis(self, sensis: pd.DataFrame) -> Dict[str, pd.Series]:
        return {'ws': sensis['RiskValue'] * sensis['RiskWeightValue']}

    def net_weighted_sensis(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Some classes can use this simple implementation but most will need to override this
        :param data: DataFrame containing the weighted sensis
        :return: the net weighted sensis
        """
        agg_cols = ['FRTBBucket', 'GroupID', 'Underlying']
        ws = data.groupby(by=agg_cols, sort=False)['ws'].sum()
        return pd.DataFrame(np.stack(ws), index=ws.index) if self.has_term_structure else ws.to_frame()

    @abstractmethod
    def rho_correlation_matrices(self, scenario: str, bucket: Union[str, None]) -> Dict:
        """
        Return a Dictionary of rho correlation matrices, suitable for use in the linear correlation algorithm.
        Each matrix has an associated set of aggregation levels, which are the keys in the Dictionary.
        If there is no aggregation level other than the bucket itself, the key should be None. For the vanilla
        case where rho depends only on the underlying and one addition axis (eg tenor), then the Dictionary would be
        {'Underlying': rho_s - rho_d, None: rho_d}, where rho_s, rho_d are the rho matrices for the same, different
        underlying, respectively.
        :param scenario: low, medium or high
        :param bucket: the bucket identifier
        :return: Dictionary of (linearised) rho correlation matrices
        """
        pass

    def calc_intra_bucket(self, ws: pd.DataFrame, scenario: str) -> pd.DataFrame:
        # Whatever else, we need the aggregated weighted sensis
        ws_matrix = ws.to_numpy()
        ws_sums = pd.Series(np.sum(ws_matrix, axis=1), index=ws.index, name='ws')

        if self.include_all:
            # Only compute the sum of the squares if we need to
            wss_sums = pd.Series(np.sum(np.square(ws_matrix), axis=1), index=ws.index, name='ssb')
            result = pd.concat([ws_sums, wss_sums], axis=1).groupby(level=['FRTBBucket', 'GroupID'], sort=False).sum()
        else:
            result = ws_sums.groupby(level=['FRTBBucket', 'GroupID'], sort=False).sum().to_frame()

        if self.bucket_dependent_rho_correlation:
            # Construct a dictionary of bucket rho correlation matrices
            buckets = ws.index.get_level_values('FRTBBucket').unique()
            rho_buckets = [bucket for bucket in buckets if bucket != self.other_bucket]
            rhos = {bucket: self.rho_correlation_matrices(scenario, bucket) for bucket in rho_buckets}

            # Determine the number of groups - this will be used to construct kb^2 full-length arrays for each bucket
            ngroups = ws.index.get_level_values('GroupID').max() + 1

            # Apply the linear algorithm for each bucket, calc_kb2 returns array which we stack into GroupID columns
            per_bucket = ws.groupby(level='FRTBBucket')
            kb2 = per_bucket.apply(self.calc_kb2, rhos=rhos, other_bucket=self.other_bucket, ngroups=ngroups)
            kb2 = pd.DataFrame(np.stack(kb2)).assign(FRTBBucket=kb2.index)

            # Normalise the DataFrame, so we have FRTBBucket, GroupID and kb^2 columns
            kb2 = pd.melt(kb2, id_vars='FRTBBucket', var_name='GroupID', value_name='kb^2')
            result['kb^2'] = kb2.set_index(keys=['FRTBBucket', 'GroupID']).loc[result.index, 'kb^2']
        else:
            # Same rho matrices across all buckets, use optimised kb2 function which exploits this
            rhos = self.rho_correlation_matrices(scenario, None)
            result['kb^2'] = self.calc_all_kb2(ws=ws, rhos=rhos, other_bucket=self.other_bucket)

        # Floor kb^2 at zero and take the square root to get the bucket risk positions
        result['kb'] = np.sqrt(result['kb^2'].clip(lower=0))

        if self.include_all:
            result['xb'] = result['kb^2'] - result['ssb']

        return result

    # noinspection PyMethodMayBeStatic
    def calculate_sb(self, kb: pd.DataFrame):
        ws, k_b = kb['ws'], kb['kb']
        kb['sb'] = ws
        kb['alt_sb'] = np.maximum(np.minimum(ws, k_b), -k_b)

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
                df = pd.DataFrame({'sb': sb, 'sb2': np.square(sb)}).groupby(level='GroupID', sort=False).sum()
                return default_gamma * (np.square(df['sb']) - df['sb2'])
            else:
                # Use explicit gamma matrix
                # Construct a matrix of the inputs to the correlation
                # Explicitly align rows with gamma buckets, columns with groups as unstack sorts index by default,
                # current version of pandas does not support the new sort=False option
                matrix = sb.unstack('GroupID', fill_value=0.0).loc[buckets, group_ids]
                return np.einsum('ji,jk,ki->i', matrix, gamma, matrix)

        kb2 = np.square(kb['kb'])
        aggregated = kb.assign(ss=kb2).groupby(level='GroupID', sort=False)[['ws', 'ss']].sum()
        ws = aggregated['ws'].to_numpy()
        ss = aggregated['ss'].to_numpy()

        # Calculate k^2, using alt sb if necessary
        k2 = ss + correlate(kb['sb'])

        alt_flag = k2 < 0

        if np.any(alt_flag):
            alt_k2 = ss + correlate(kb['alt_sb'])
            k2[k2 < 0] = alt_k2[k2 < 0]

        x = k2 - ss
        k = np.where(k2 > 0, np.sqrt(k2), 0.0)

        index = pd.Index(group_ids, name='GroupID')

        return pd.DataFrame({'ws': ws, 'k': k, 'ss': ss, 'x': x, 'alt_flag': alt_flag.astype(np.int8)}, index=index)

    @staticmethod
    def calc_kb2(ws: pd.DataFrame, rhos: Dict, ngroups: int, other_bucket=None) -> np.array:
        """
        Compute the square of the risk position for a given bucket
        :param ws: net weighted sensis for a given bucket
        :param rhos: dictionary of rho correlations with bucket as key
        :param ngroups: the total number of groups (across all buckets, not just this one)
        :param other_bucket: optional, specifies the 'other' bucket and is treated separately
        :return: kb^2 for all groups, in a numpy array
        """

        # Determine the bucket, which we will use to resolve the rho matrix (unless it is the 'Other' bucket)
        bucket = ws.index.get_level_values('FRTBBucket')[0]

        # We can dispense with the bucket index now, leaving the GroupID index + any other axes
        ws = ws.droplevel(level='FRTBBucket')

        # Extract the unique list of GroupId values
        group_ids = ws.index.get_level_values('GroupID').unique()

        # Running totals of ALL group K_b^2 values, initialised to zero
        total = np.repeat(0.0, ngroups)

        if bucket == other_bucket:
            other = ws.groupby(level='GroupID', sort=False).apply(lambda x: np.square(np.sum(np.abs(x.to_numpy()))))
            total[group_ids] += other.to_numpy()
        else:
            for level, rho in rhos[bucket].items():
                # Aggregate/net risk at specified level(s) (bucket level is None, so level will be 'GroupID' only
                level = ['GroupID', level] if isinstance(level, str) else ['GroupID', *(level or [])]

                # Check if we can avoid having to do an unnecessary sum() ie data already aggregated at required level
                aggregation_required = ws.index.nlevels > len(level)

                # Construct matrix of net weighted sensis and compute the diagonal of the WCW^T matrix multiplication
                ws_level = ws.groupby(level=level, sort=False).sum() if aggregation_required else ws
                ws_matrix = ws_level.to_numpy()
                diagonal = np.einsum('ij,jk,ik->i', ws_matrix, rho, ws_matrix)

                if len(level) > 1:
                    # We need to take the sum of the diagonal values for each group
                    groups = pd.Series(diagonal).groupby(by=ws_level.index.get_level_values('GroupID'), sort=False)
                    total[group_ids] += groups.sum().to_numpy()
                else:
                    # level is GroupID only, just read off the diagonal values (one per group)
                    total[group_ids] += diagonal

        return total

    @staticmethod
    def calc_all_kb2(ws: pd.DataFrame, rhos: Dict, other_bucket=None) -> pd.Series:
        """
        Compute the square of the risk position for ALL buckets in one batch
        :param ws: net weighted sensis for ALL buckets
        :param rhos: dictionary of rho correlations per level (common to all buckets)
        :param other_bucket: optional, specifies the 'other' bucket and is treated separately
        :return: kb^2 Series, indexed by FRTBBucket, GroupID
        """
        total = None

        for level, rho in rhos.items():
            # We need to aggregate/net risk at FRTBBucket and GroupID + specified sub-level(s)
            base_levels = ['FRTBBucket', 'GroupID']
            level = [*base_levels, level] if isinstance(level, str) else [*base_levels, *(level or [])]

            # Check if we can avoid having to do an unnecessary sum() ie data already aggregated at required level
            aggregation_required = ws.index.nlevels > len(level)

            # Construct matrix of net weighted sensis and compute the diagonal of the WCW^T matrix multiplication
            ws_level = ws.groupby(level=level, sort=False).sum() if aggregation_required else ws
            ws_matrix = ws_level.to_numpy()
            diagonal = np.einsum('ij,jk,ik->i', ws_matrix, rho, ws_matrix)

            contribution = pd.Series(diagonal, index=ws_level.index)

            if len(level) > 2:
                contribution = contribution.groupby(level=['FRTBBucket', 'GroupID']).sum()

            total = contribution if total is None else total + contribution

        if other_bucket is not None and other_bucket in ws.index:
            # Override the other bucket kb2 using the alternative prescription
            other = ws.loc[other_bucket].groupby(level='GroupID', sort=False)
            total.loc[other_bucket].loc[:] = other.apply(lambda x: np.square(np.sum(np.abs(x.to_numpy()))))

        return total


class VegaCalculator(DeltaVegaCalculator):
    """
    Base class for Vega Calculators
    """

    # Vega always has term structure ie the option maturity dates
    has_term_structure = True

    def rho_correlation_matrices(self, scenario: str, bucket: Union[str, None]) -> Dict:
        """
        Default rho correlation matrices - tenor-tenor correlation for same/different underlying name.
        Suitable for use in the linear correlation algorithm.
        :param scenario: low, medium or high
        :param bucket: the bucket identifier
        :return: Dictionary of rho correlation matrices, for underlying and bucket aggregation levels
        """
        name, tenor1, tenor2 = 'Underlying1Value', 'Underlying2Value', 'Underlying3Value'
        rhos = self.rho_correlations(bucket=bucket, rows=(name, tenor1), columns=tenor2)
        rho = rhos[scenario]

        index = self.params['BucketCode']
        rho_s = rho.loc['Same'].loc[index, index].to_numpy()
        rho_d = rho.loc['Different'].loc[index, index].to_numpy()

        return {
            'Underlying': rho_s - rho_d,
            None: rho_d
        }
