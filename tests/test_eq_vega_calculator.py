import pytest

from tranql.models.frtb.sa.calculator.vega_calculators import EQVegaCalculator


@pytest.fixture
def calculator(calculator_factory):
    bucket_codes = ['0.5Y', '1Y', '3Y', '5Y', '10Y']
    return calculator_factory(EQVegaCalculator, params={'BucketCode': bucket_codes})


@pytest.mark.risk_type('EQVega')
@pytest.mark.parametrize('scenario', ['low', 'medium', 'high'])
@pytest.mark.parametrize('bucket', range(1, 14))
class TestEQVegaCalculator:
    def test_rho_correlation_matrices(self, calculator: EQVegaCalculator, scenario: str, bucket: str):
        rho = calculator.rho_correlation_matrices(scenario=scenario, bucket=str(bucket))

        assert rho is not None
        assert len(rho) == 2

        assert 'Underlying' in rho
        assert rho['Underlying'].shape == (5, 5)

        assert None in rho
        assert rho[None].shape == (5, 5)
