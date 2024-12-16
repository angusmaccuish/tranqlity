import pytest

from tranql.models.frtb.sa.calculator.delta_calculators import CSRSecDeltaCalculator


@pytest.fixture
def calculator(calculator_factory):
    bucket_codes = ['0.5Y', '1Y', '3Y', '5Y', '10Y']
    return calculator_factory(CSRSecDeltaCalculator, params={'BucketCode': bucket_codes})


@pytest.mark.risk_type('CSRSecDelta')
@pytest.mark.parametrize('scenario', ['low', 'medium', 'high'])
class TestCSRSecDeltaCalculator:
    def test_rho_correlation_matrices(self, calculator: CSRSecDeltaCalculator, scenario: str):
        rho = calculator.rho_correlation_matrices(scenario=scenario, bucket=None)

        assert rho is not None
        assert len(rho) == 2

        assert 'Underlying' in rho
        assert rho['Underlying'].shape == (10, 10)

        assert None in rho
        assert rho[None].shape == (10, 10)
