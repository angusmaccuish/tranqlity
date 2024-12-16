from typing import Dict, Union

import numpy as np
import pandas as pd

from .delta_vega_calculator import VegaCalculator
from .sbm_calculator import OtherBucketOverride


class CMVegaCalculator(VegaCalculator):
    bucket_dependent_rho_correlation = True


class CSRVegaCalculator(VegaCalculator):
    bucket_dependent_rho_correlation = True
    other_bucket_property = 'sa.csrnonsec.otherSector'


class CSRSecVegaCalculator(OtherBucketOverride, VegaCalculator):
    bucket_dependent_rho_correlation = True
    other_bucket_property = 'sa.csrsec.otherSector'


class EQVegaCalculator(VegaCalculator):
    bucket_dependent_rho_correlation = True
    other_bucket_property = 'sa.equity.otherSector'


class FXVegaCalculator(VegaCalculator):
    """
    FX Vega intra-bucket correlation has neither bucket nor Underlying dependency, so we can just aggregate the
    weighted risk at the bucket level. The linear algorithm reduces to the simple, bucket-level only case.

    Note that there is no 'Other' bucket.
    """

    bucket_dependent_rho_correlation = False

    def net_weighted_sensis(self, data: pd.DataFrame) -> pd.DataFrame:
        ws = data.groupby(by=['FRTBBucket', 'GroupID'], sort=False)['ws'].sum()
        return pd.DataFrame(np.stack(ws.to_numpy()), index=ws.index)

    def rho_correlation_matrices(self, scenario: str, bucket: Union[str, None]) -> Dict:
        tenor1, tenor2 = 'Underlying1Value', 'Underlying2Value'
        rhos = self.rho_correlations(bucket=bucket, rows=tenor1, columns=tenor2)
        rho = rhos[scenario]

        index = self.params['BucketCode']

        return {
            None: rho.loc[index, index].to_numpy()
        }


class IRVegaCalculator(VegaCalculator):
    """
    GIRR Vega, like its Delta counterpart, has a non-trivial risk-weighted sensitivity de-normalization, making it the
    most complex calculator to implement. As with the Delta case, we denormalize the YIELD, BASIS and INFLATION risk,
    however now there are two sets of tenors: the option maturity tenors and the underlying maturity tenors. Again, the
    BASIS and INFLATION risk is rolled up, but only across the underlying maturity tenors, leaving a risk value per
    option maturity tenor. So if there are M underlying maturity tenors and N option maturity tenors, we have MxN YIELD
    values, N BASIS values and N INFLATION values, which means we must construct a vector of size MxN+N+N = (M+2)xN.

    Naturally, our rho correlation matrix must be of dimension (M+2)xN in order to be compatible with the above.

    Note that the rho correlation matrices are bucket-independent, which allows for an additional optimisation as we
    don't need to explicitly split the risk into buckets. There is no 'Other' bucket.
    """

    bucket_dependent_rho_correlation = False

    def net_weighted_sensis(self, data: pd.DataFrame) -> pd.DataFrame:
        # Denormalize so that we have YIELD, BASIS and INFLATION (RiskFactorType) top-level columns
        # with underlying maturity (BucketCode2) as nested columns
        option_maturities = len(self.params['BucketCode'])
        underlying_maturities = len(self.params['BucketCode2'])
        agg_cols = ['FRTBBucket', 'GroupID', 'Underlying', 'BucketCode2', 'RiskFactorType']
        df = data.groupby(by=agg_cols, sort=False)['ws'].sum()
        df = df.unstack(level=['RiskFactorType', 'BucketCode2'], fill_value=np.zeros(option_maturities))

        out = np.zeros((len(df), option_maturities*(underlying_maturities+2)))

        # YIELD just a straight copy, BASIS/INFLATION roll up the underlying maturity vega risk
        if 'YIELD' in df.columns:
            for k, underlying_maturity in enumerate(self.params['BucketCode2']):
                start = k*option_maturities
                np.stack(df['YIELD'][underlying_maturity], out=out[:, start:start+option_maturities])

        if 'BASIS' in df.columns:
            start = option_maturities*underlying_maturities
            np.stack(df['BASIS'].sum(axis=1), out=out[:, start:start+option_maturities])

        if 'INFLATION' in df.columns:
            start = option_maturities*(underlying_maturities+1)
            np.stack(df['INFLATION'].sum(axis=1), out=out[:, start:start+option_maturities])

        return pd.DataFrame(out, index=df.index)

    def rho_correlation_matrices(self, scenario: str, bucket: Union[str, None]) -> Dict:
        yield_index = [(bc2, bc) for bc2 in self.params['BucketCode2'] for bc in self.params['BucketCode']]
        basis_index = [('BASIS', bc) for bc in self.params['BucketCode']]
        inflation_index = [('INFLATION', bc) for bc in self.params['BucketCode']]
        index = [*yield_index, *basis_index, *inflation_index]

        underlying_maturity1, option_maturity1 = 'Underlying3Value', 'Underlying1Value'
        underlying_maturity2, option_maturity2 = 'Underlying4Value', 'Underlying2Value'
        rows, columns = (underlying_maturity1, option_maturity1), (underlying_maturity2, option_maturity2)
        rhos = self.rho_correlations(bucket=bucket, rows=rows, columns=columns)
        rho = rhos[scenario]

        return {
            None: rho.loc[index, index].to_numpy()
        }
