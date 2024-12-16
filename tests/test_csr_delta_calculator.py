import pytest

from tranql.models.frtb.sa.calculator.delta_calculators import CSRDeltaCalculator


@pytest.fixture
def calculator(calculator_factory):
    bucket_codes = ['0.5Y', '1Y', '3Y', '5Y', '10Y']
    return calculator_factory(CSRDeltaCalculator, params={'BucketCode': bucket_codes})


@pytest.mark.risk_type('CSRDelta')
@pytest.mark.parametrize('scenario', ['low', 'medium', 'high'])
class TestCSRDeltaCalculator:
    def test_rho_correlation_matrices(self, calculator: CSRDeltaCalculator, scenario: str):
        rho = calculator.rho_correlation_matrices(scenario=scenario, bucket=None)

        assert rho is not None
        assert len(rho) == 2

        assert 'Underlying' in rho
        assert rho['Underlying'].shape == (10, 10)

        assert None in rho
        assert rho[None].shape == (10, 10)
