import pandas as pd
import pytest

from tranql.models.frtb.sa.calculator.drc_calculator import DRCNonSecCalculator


@pytest.fixture
def calculator(calculator_factory):
    return calculator_factory(DRCNonSecCalculator)


@pytest.mark.risk_type('JTD')
class TestDRCNonSecCalculator:
    def test_calculate_levels(self, calculator: DRCNonSecCalculator):
        sensis = pd.DataFrame({
            'GroupID': [1],
            'FRTBBucket': ['corporates'],
            'ObligorId': ['ObligorA'],
            'RWLabel': ['corporates_A'],
            'DebtSeniority': ['SENIOR'],
            'ShareType': ['NOTFUND'],
            'ScaledGrossJTD': [100.0],
        })

        result = calculator.calculate_levels(sensis=sensis)

        assert result is not None
        assert len(result) == 2
        assert 'total' in result
        assert 'bucket' in result
