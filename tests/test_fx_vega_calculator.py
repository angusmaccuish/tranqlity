import pytest

from tranql.models.frtb.sa.calculator.vega_calculators import FXVegaCalculator


@pytest.fixture
def calculator(calculator_factory):
    bucket_codes = ['0.5Y', '1Y', '3Y', '5Y', '10Y']
    return calculator_factory(FXVegaCalculator, params={'BucketCode': bucket_codes})


@pytest.mark.risk_type('FXVega')
@pytest.mark.parametrize('scenario', ['low', 'medium', 'high'])
class TestFXVegaCalculator:
    def test_rho_correlation_matrices(self, calculator: FXVegaCalculator, scenario: str):
        rho = calculator.rho_correlation_matrices(scenario=scenario, bucket=None)

        assert rho is not None
        assert len(rho) == 1
        assert None in rho
        assert rho[None].shape == (5, 5)
