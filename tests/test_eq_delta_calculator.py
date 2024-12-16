import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tranql.models.frtb.sa.calculator.delta_calculators import EQDeltaCalculator


@pytest.fixture
def calculator(calculator_factory):
    return calculator_factory(EQDeltaCalculator)


@pytest.mark.risk_type('EQDelta')
class TestEQDeltaCalculator:
    @pytest.mark.parametrize('scenario', ['low', 'medium', 'high'])
    @pytest.mark.parametrize('bucket', range(1, 14))
    def test_rho_correlation_matrices(self, calculator: EQDeltaCalculator, scenario: str, bucket: str):
        rho = calculator.rho_correlation_matrices(scenario=scenario, bucket=str(bucket))

        assert rho is not None
        assert len(rho) == 2

        assert 'Underlying' in rho
        assert rho['Underlying'].shape == (2, 2)

        assert None in rho
        assert rho[None].shape == (2, 2)

    def test_net_weighted_sensis(self, calculator: EQDeltaCalculator):
        sensis = pd.DataFrame({
            'GroupID': [1, 1, 1, 2, 2],
            'FRTBBucket': ['USD', 'USD', 'GBP', 'EUR', 'CHF'],
            'Underlying': ['USD', 'USD', 'GBP', 'EUR', 'CHF'],
            'RiskFactorType': ['SPOT', 'REPO', 'SPOT', 'SPOT', 'SPOT'],
            'ws': [10.0, 20.0, 30.0, 40.0, 50.0],
        })

        net_weighted_sensis = calculator.net_weighted_sensis(data=sensis)

        expected = pd.DataFrame({
            'FRTBBucket': ['USD', 'GBP', 'EUR', 'CHF'],
            'GroupID': [1, 1, 2, 2],
            'Underlying': ['USD', 'GBP', 'EUR', 'CHF'],
            'SPOT': [10.0, 30.0, 40.0, 50.0],
            'REPO': [20.0, 0.0, 0.0, 0.0],
        })

        assert_frame_equal(left=expected.set_index(['FRTBBucket', 'GroupID', 'Underlying']),
                           right=net_weighted_sensis,
                           check_like=True,
                           check_names=False)
