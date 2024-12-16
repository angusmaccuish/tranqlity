from typing import Dict

import numpy as np
import pandas as pd

from .curvature_calculator import CurvatureCalculator, SimpleCurvatureCalculator
from .sbm_calculator import OtherBucketOverride


class CMCurvatureCalculator(CurvatureCalculator):
    pass


class CSRCurvatureCalculator(CurvatureCalculator):
    other_bucket_property = 'sa.csrnonsec.otherSector'


class CSRSecCurvatureCalculator(OtherBucketOverride, CurvatureCalculator):
    other_bucket_property = 'sa.csrsec.otherSector'


class EQCurvatureCalculator(CurvatureCalculator):
    other_bucket_property = 'sa.equity.otherSector'


class FXCurvatureCalculator(SimpleCurvatureCalculator):
    """
    FX Curvature has one subtlety which requires some override logic: options which do not reference the reporting
    currency can have their CVR divided by 1.5. Such risk is identified by checking the 'FXDivisorEligibility' column,
    and the actual divisor is determined using the 'sa.fx.divider.value' parameter, which allows for variations across
    different regulatory regimes.
    """

    def get_weighted_sensis(self, sensis: pd.DataFrame) -> Dict[str, pd.Series]:
        ws = super().get_weighted_sensis(sensis)

        if self.params['sa.fx.divider.enabled'].upper() == 'Y':
            fx_divisor = float(self.params['sa.fx.divider.value'])
            fx_divisor_eligible = sensis['FXDivisorEligibility'] == 'Y'
            return {
                'cvru': np.where(fx_divisor_eligible, ws['cvru'] / fx_divisor, ws['cvru']),
                'cvrd': np.where(fx_divisor_eligible, ws['cvrd'] / fx_divisor, ws['cvrd']),
                'ws': ws['ws'],
            }
        else:
            return ws


class IRCurvatureCalculator(SimpleCurvatureCalculator):
    pass
