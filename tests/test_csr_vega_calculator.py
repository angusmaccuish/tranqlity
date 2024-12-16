import pytest

from tranql.models.frtb.sa.calculator.vega_calculators import CSRVegaCalculator


@pytest.fixture
def calculator(calculator_factory):
    bucket_codes = ['0.5Y', '1Y', '3Y', '5Y', '10Y']
    return calculator_factory(CSRVegaCalculator, params={'BucketCode': bucket_codes})


@pytest.mark.regulation('crr2')
@pytest.mark.risk_type('CSRVega')
@pytest.mark.parametrize('scenario', ['low', 'medium', 'high'])
class TestCSRVegaCalculator:
    def test_rho_correlation_matrices(self, calculator: CSRVegaCalculator, scenario: str):
        rho = calculator.rho_correlation_matrices(scenario=scenario, bucket=None)

        assert rho is not None
        assert len(rho) == 2

        assert 'Underlying' in rho
        assert rho['Underlying'].shape == (5, 5)

        assert None in rho
        assert rho[None].shape == (5, 5)
