from typing import Dict, Optional

import pandas as pd

from tranql.models.frtb.sa.config.regulation_params import reg_params
from .calculator_mapping import calculator_mapping


def calculate(sensis: pd.DataFrame,
              reg_name: Optional[str] = 'd457',
              risk_weights: Optional[pd.DataFrame] = pd.DataFrame(),
              correlations: Optional[pd.DataFrame] = pd.DataFrame(),
              run_params: Optional[Dict] = None,
              calc_params: Optional[pd.DataFrame] = pd.DataFrame(),
              seniority_ranking: Optional[pd.DataFrame] = pd.DataFrame(),
              charge_only: Optional[bool] = False) -> Dict | pd.DataFrame:

    # CalcName parameter used to find appropriate calculator
    calculator_name = run_params.get('CalcName')

    # Override standard regulatory parameters with additional parameters.
    if run_params is not None:
        params = {**reg_params[reg_name], **run_params}
    else:
        params = reg_params[reg_name]

    if calc_params is not None and not calc_params.empty:
        params.update(calc_params.set_index('ParameterName')['ParameterValue'].to_dict())

    # Get relevant calculator
    cls = calculator_mapping[calculator_name]

    calculator = cls(reg_name=reg_name,
                     risk_weights=risk_weights,
                     correlations=correlations,
                     params=params,
                     seniority_ranking=seniority_ranking)

    if charge_only:
        return calculator.calculate_charge(sensis)
    else:
        return calculator.calculate(sensis)
