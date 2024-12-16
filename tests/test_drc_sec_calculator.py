import pandas as pd
import pytest

from tranql.models.frtb.sa.calculator.drc_calculator import DRCSecCalculator


@pytest.fixture
def calculator(calculator_factory):
    return calculator_factory(DRCSecCalculator)


@pytest.mark.risk_type('DRCSec')
class TestDRCSecCalculator:
    def test_calculate_levels(self, calculator: DRCSecCalculator):
        sensis = pd.DataFrame({
            'GroupID': [1],
            'FRTBBucket': ['Bucket1'],
            'FRTBDRCTranche': ['TrancheA'],
            'DRCSecRiskWeight': [0.1],
            'ScaledGrossJTD': [1000.0],
        })

        result = calculator.calculate_levels(sensis=sensis)

        assert result is not None
        assert len(result) == 2
        assert 'total' in result
        assert 'bucket' in result
