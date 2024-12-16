from typing import Dict

import pandas as pd

from .calculator import Calculator


class RRAOCalculator(Calculator):
    def calculate_levels(self, sensis: pd.DataFrame) -> Dict:
        # Apply risk weights
        risk_weights = sensis['RWLabel'].map(self.risk_weights)
        sensis['ws'] = sensis['AbsoluteRiskValue'] * risk_weights

        # there are no buckets, just return the 'total' DataFrame
        aggregations = {'RRAO.Notional': ('AbsoluteRiskValue', 'sum'), 'RRAO': ('ws', 'sum')}
        total_df = sensis.groupby(by='GroupID', sort=False).agg(**aggregations)
        return {'total': total_df}
