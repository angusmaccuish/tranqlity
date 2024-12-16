import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tranql.models.frtb.sa.calculator.delta_calculators import FXDeltaCalculator


@pytest.fixture
def calculator(calculator_factory):
    return calculator_factory(FXDeltaCalculator)


# noinspection SpellCheckingInspection
@pytest.mark.risk_type('FXDelta')
class TestFXDeltaCalculator:
    @pytest.mark.parametrize('scenario', ['low', 'medium', 'high'])
    def test_rho_correlation_matrices(self, calculator: FXDeltaCalculator, scenario: str):
        rho = calculator.rho_correlation_matrices(scenario=scenario, bucket=None)

        assert rho is None

    def test_net_weighted_sensis(self, calculator: FXDeltaCalculator):
        sensis = pd.DataFrame({
            'GroupID': [1, 1, 1, 2, 2],
            'FRTBBucket': ['USDGBP', 'USDGBP', 'EURGBP', 'TWDUSD', 'CHFUSD'],
            'ws': [10.0, 10.0, 30.0, 100.00, 200.0]
        })

        net_weighted_sensis = calculator.net_weighted_sensis(data=sensis)

        expected = pd.DataFrame({
            'GroupID': [1, 1, 2, 2],
            'FRTBBucket': ['USDGBP', 'EURGBP', 'TWDUSD', 'CHFUSD'],
            'ws': [20.0, 30.0, 100.0, 200.0]
        })

        assert_frame_equal(left=expected.set_index(['FRTBBucket', 'GroupID']),
                           right=net_weighted_sensis,
                           check_like=True,
                           check_names=False)
