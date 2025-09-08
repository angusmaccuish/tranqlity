from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd

from tranql.models.frtb.sa.config.regulation_params import reg_params
from .utils import explode_bucket_code, handle_risk_weight_measure, reindex_group_ids


class Calculator(ABC):
    """
    Base class for all Calculators
    """

    def __init__(self,
                 reg_name: str,
                 risk_weights: pd.DataFrame,
                 correlations: pd.DataFrame,
                 seniority_ranking: pd.DataFrame,
                 params: Dict):
        # For regulator-specific logic
        self.reg_name = reg_name.lower()

        # Convert risk weights DataFrame to a dictionary: RWLabel -> Risk Weight
        if risk_weights is not None and not risk_weights.empty:
            self.risk_weights = risk_weights.set_index('RWLabel')['RiskWeightValue'].to_dict()
        else:
            self.risk_weights = None

        self.correlations = correlations
        self.seniority_ranking = seniority_ranking

        if params is not None:
            self.params = {**reg_params[reg_name], **params}
        else:
            self.params = reg_params[reg_name]

    @explode_bucket_code
    @reindex_group_ids
    @handle_risk_weight_measure
    def calculate(self, sensis: pd.DataFrame, params: Optional[dict] = None) -> dict:
        # Overwrite params
        if params is not None:
            self.params = {**self.params, **params}

        # Perform calculation
        results = self.calculate_levels(sensis)

        # Convert results to correct format
        return self.reduce(results)

    def calculate_charge(self, sensis: pd.DataFrame) -> pd.DataFrame:
        result = self.calculate(sensis, params={'level': ['total']})
        return result['total']

    @abstractmethod
    def calculate_levels(self, sensis: pd.DataFrame) -> Dict:
        """
        This is where virtually all the computation is done
        :param sensis: the input DataFrame
        :return: Dictionary of results, DataFrames keyed on level ('total' and/or 'bucket')
        """
        pass

    def reduce(self, results: Dict):
        """
        Only return level(s) which have been actually requested (it is often just as quick to compute everything).
        Reset the index on the DataFrame(s) so that FRTBBucket/GroupID appears as a column.
        :param results: the Dictionary of results
        :return: (possibly filtered) Dictionary of results, with any DataFrame index reset
        """
        return {level: df.reset_index() for level, df in results.items() if level in self.params['level']}
