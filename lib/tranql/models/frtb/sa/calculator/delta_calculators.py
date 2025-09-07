from typing import Dict, Optional

import numpy as np
import pandas as pd

from .delta_vega_calculator import DeltaVegaCalculator
from .sbm_calculator import OtherBucketOverride


class CMDeltaCalculator(DeltaVegaCalculator):
    """
    Commodity Delta is the one risk class/type combination which requires an extra nested level: CMTYLocation.
    So the usual tenor-tenor correlation now has to consider same/different Underlying with same/different CMTYLocation.
    This requires 4 matrices in the linear algorithm, for the 4 different levels of aggregation:

      * Underlying, CMTYLocation
      * Underlying
      * CMTYLocation
      * Bucket-level

    Note that these correlation matrices are bucket-dependent, and that there is no Commodity 'Other' bucket.
    """

    bucket_dependent_rho_correlation = True
    has_term_structure = True

    def net_weighted_sensis(self, data: pd.DataFrame) -> pd.DataFrame:
        agg_cols = ['FRTBBucket', 'GroupID', 'Underlying', 'CMTYLocation']
        ws = data.groupby(by=agg_cols, sort=False)['ws'].sum()
        return pd.DataFrame(np.stack(ws), index=ws.index)

    def rho_correlation_matrices(self, scenario: str, bucket: Optional[str]) -> Dict:
        name, tenor, location = 'Underlying1Value', 'Underlying2Value', 'Underlying3Value'
        rhos = self.rho_correlations(bucket=bucket, rows=(name, tenor), columns=location)
        rho = rhos[scenario]

        shape = (len(self.params['BucketCode']), len(self.params['BucketCode']))

        # Fill value is the off-diagonal different tenor value (name/location = same/same)
        rho_ss = np.full(shape, fill_value=rho.loc[('Same', 'Different'), 'Same'])
        np.fill_diagonal(rho_ss, val=rho.loc[('Same', 'Same'), 'Same'])

        # Fill value is the off-diagonal different tenor value (name/location = same/different)
        rho_sd = np.full(shape, fill_value=rho.loc[('Same', 'Different'), 'Different'])
        np.fill_diagonal(rho_sd, val=rho.loc[('Same', 'Same'), 'Different'])

        # Fill value is the off-diagonal different tenor value (name/location = different/same)
        rho_ds = np.full(shape, fill_value=rho.loc[('Different', 'Different'), 'Same'])
        np.fill_diagonal(rho_ds, val=rho.loc[('Different', 'Same'), 'Same'])

        # Fill value is the off-diagonal different tenor value (name/location = different/different)
        rho_dd = np.full(shape, fill_value=rho.loc[('Different', 'Different'), 'Different'])
        np.fill_diagonal(rho_dd, val=rho.loc[('Different', 'Same'), 'Different'])

        return {
            ('Underlying', 'CMTYLocation'): rho_ss - rho_ds - rho_sd + rho_dd,
            'Underlying': rho_sd - rho_dd,
            'CMTYLocation': rho_ds - rho_dd,
            None: rho_dd
        }


class CSRDeltaCalculator(DeltaVegaCalculator):
    """
    CSR Delta does have an additional correlation axis (the basis, 'BOND'/'CDS'), however it is more convenient/faster
    to denormalize the weighted risk along this axis instead of using the 4 matrix approach (see Commodity Delta above).
    Each underlying risk weighted sensitivity vector will be a concatenation of the BOND and CDS tenor risk, with the
    correlation matrices constructed to be compatible with this denormalized form.

    Note that the rho correlation matrices are bucket-dependent, and that there is an 'Other' bucket.
    """

    bucket_dependent_rho_correlation = True
    has_term_structure = True
    other_bucket_property = 'sa.csrnonsec.otherSector'

    def net_weighted_sensis(self, data: pd.DataFrame) -> pd.DataFrame:
        buckets = len(self.params['BucketCode'])
        agg_cols = ['FRTBBucket', 'GroupID', 'Underlying', 'RiskFactorType']
        df = data.groupby(by=agg_cols, sort=False)['ws'].sum()
        df = df.unstack(level='RiskFactorType', fill_value=np.zeros(buckets))

        out = np.zeros((len(df), 2*buckets))

        if 'BOND' in df.columns:
            np.stack(df['BOND'], out=out[:, :buckets])

        if 'CDS' in df.columns:
            np.stack(df['CDS'], out=out[:, buckets:])

        return pd.DataFrame(out, index=df.index)

    def rho_correlation_matrices(self, scenario: str, bucket: Optional[str]) -> Dict:
        name, tenor, basis = 'Underlying1Value', 'Underlying2Value', 'Underlying3Value'
        rhos = self.rho_correlations(bucket=bucket, rows=(name, tenor), columns=basis)
        rho = rhos[scenario]

        dimension = len(self.params['BucketCode'])
        shape = (dimension, dimension)

        # Fill value is the off-diagonal different tenor value (name/basis = same/same)
        rho_ss = np.full(shape, fill_value=rho.loc[('Same', 'Different'), 'Same'])
        np.fill_diagonal(rho_ss, val=rho.loc[('Same', 'Same'), 'Same'])

        # Fill value is the off-diagonal different tenor value (name/basis = same/different)
        rho_sd = np.full(shape, fill_value=rho.loc[('Same', 'Different'), 'Different'])
        np.fill_diagonal(rho_sd, val=rho.loc[('Same', 'Same'), 'Different'])

        # Fill value is the off-diagonal different tenor value (name/basis = different/same)
        rho_ds = np.full(shape, fill_value=rho.loc[('Different', 'Different'), 'Same'])
        np.fill_diagonal(rho_ds, val=rho.loc[('Different', 'Same'), 'Same'])

        # Fill value is the off-diagonal different tenor value (name/basis = different/different)
        rho_dd = np.full(shape, fill_value=rho.loc[('Different', 'Different'), 'Different'])
        np.fill_diagonal(rho_dd, val=rho.loc[('Different', 'Same'), 'Different'])

        rho_s = np.block([[rho_ss, rho_sd], [rho_sd, rho_ss]])
        rho_d = np.block([[rho_ds, rho_dd], [rho_dd, rho_ds]])

        return {
            'Underlying': rho_s - rho_d,
            None: rho_d
        }


class CSRSecDeltaCalculator(OtherBucketOverride, CSRDeltaCalculator):
    """
    CSR Securitized non-CTP Delta shares the same intra-bucket correlation logic as the non securitized version, however
    there is an additional subtlety when considering the inter-bucket correlation.

    Note that there is an 'Other' bucket.
    """

    other_bucket_property = 'sa.csrsec.otherSector'


class EQDeltaCalculator(DeltaVegaCalculator):
    """
    Equity Delta, like CSR, has an additional correlation axis which we can denormalize: the basis, 'SPOT'/'REPO'.
    There are no tenors, so the rho matrices boil down to 2x2 dimensions for same/different Underlying with
    same/different basis.

    Note that the rho correlation matrices are bucket-dependent, and that there is an 'Other' bucket.
    """

    bucket_dependent_rho_correlation = True
    has_term_structure = False
    other_bucket_property = 'sa.equity.otherSector'

    def net_weighted_sensis(self, data: pd.DataFrame) -> pd.DataFrame:
        agg_cols = ['FRTBBucket', 'GroupID', 'Underlying', 'RiskFactorType']
        ws = data.groupby(by=agg_cols, sort=False)['ws'].sum()
        df = ws.unstack(level='RiskFactorType', fill_value=0.0)
        return df.reindex(columns=['SPOT', 'REPO'], fill_value=0.0)

    def rho_correlation_matrices(self, scenario: str, bucket: Optional[str]) -> Dict:
        name, basis = 'Underlying1Value', 'Underlying2Value'
        rhos = self.rho_correlations(bucket=bucket, rows=name, columns=basis)
        rho = rhos[scenario]

        rho_s, rho_d = [
            np.array([
                [rho.loc[(underlying, 'Same')], rho.loc[(underlying, 'Different')]],
                [rho.loc[(underlying, 'Different')], rho.loc[(underlying, 'Same')]]
            ]) for underlying in ('Same', 'Different')
        ]

        return {
            'Underlying': rho_s - rho_d,
            None: rho_d
        }


class FXDeltaCalculator(DeltaVegaCalculator):
    """
    FX Delta is particularly simple, given that there is no tenor structure and the bucket contains a single underlying.
    This means there is no rho cross-correlation contribution, leaving a trivial bucket risk-position calculation.

    Note that there is no 'Other' bucket.
    """

    bucket_dependent_rho_correlation = False
    has_term_structure = False

    def net_weighted_sensis(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.groupby(by=['FRTBBucket', 'GroupID'], sort=False)[['ws']].sum()

    def rho_correlation_matrices(self, scenario: str, bucket: Optional[str]) -> Dict:
        pass

    def calc_intra_bucket(self, ws: pd.DataFrame, scenario: str) -> pd.DataFrame:
        result = ws.copy()
        result['kb'] = result['ws'].abs()
        result['kb^2'] = np.square(result['kb'])

        if self.include_all:
            result['ssb'] = result['kb^2']
            result['xb'] = 0.0

        return result


class IRDeltaCalculator(DeltaVegaCalculator):
    """
    GIRR Delta is the most complicated of all the calculators in this module, although the principles are the same.
    The additional complexity comes in the form of a non-trivial de-normalization of the risk-weighted sensitivities.
    Once again we have an additional axis, the risk factor type - 'YIELD', 'BASIS' and 'INFLATION'. However, when we
    de-normalize the BASIS and INFLATION risk, we flatten them into scalar values by summing across all the tenors.
    So if there are N tenors, our de-normalized vector will have N+2 components: N YIELD tenor risk values, followed by
    the flattened BASIS and INFLATION risk.

    Naturally, our rho correlation matrices must be of dimension N+2 in order to be compatible with the above.

    Note that the rho correlation matrices are bucket-independent, which allows for an additional optimisation as we
    don't need to explicitly split the risk into buckets. There is no 'Other' bucket.
    """

    bucket_dependent_rho_correlation = False
    has_term_structure = True

    def net_weighted_sensis(self, data: pd.DataFrame) -> pd.DataFrame:
        # Denormalize so that we have YIELD, BASIS and INFLATION columns instead of rows
        buckets = len(self.params['BucketCode'])
        df = data.groupby(by=['FRTBBucket', 'GroupID', 'Underlying', 'RiskFactorType'], sort=False)['ws'].sum()
        df = df.unstack(level='RiskFactorType', fill_value=np.zeros(buckets))

        out = np.zeros((len(df), buckets+2))

        if 'YIELD' in df.columns:
            np.stack(df['YIELD'], out=out[:, :buckets])

        if 'BASIS' in df.columns:
            np.sum(np.stack(df['BASIS']), axis=1, out=out[:, buckets])

        if 'INFLATION' in df.columns:
            np.sum(np.stack(df['INFLATION']), axis=1, out=out[:, buckets+1])

        return pd.DataFrame(out, index=df.index)

    def rho_correlation_matrices(self, scenario: str, bucket: Optional[str]) -> Dict:
        name, tenor, risk_factor_type = 'Underlying1Value', 'Underlying2Value', 'Underlying3Value'
        rhos = self.rho_correlations(bucket=bucket, rows=(name, tenor), columns=risk_factor_type, fill_value=0.0)
        rho = rhos[scenario]

        index = [*self.params['BucketCode'], 'BASIS', 'INFLATION']
        rho_s, rho_d = [rho.loc[category].loc[index, index].to_numpy() for category in ('Same', 'Different')]

        return {
            'Underlying': rho_s - rho_d,
            None: rho_d
        }
