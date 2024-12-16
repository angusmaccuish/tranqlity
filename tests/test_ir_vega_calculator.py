import pytest

from tranql.models.frtb.sa.calculator.vega_calculators import IRVegaCalculator


@pytest.fixture
def calculator(calculator_factory):
    bucket_codes = bucket_code2s = ['0.5Y', '1Y', '3Y', '5Y', '10Y']
    return calculator_factory(IRVegaCalculator, params={'BucketCode': bucket_codes, 'BucketCode2': bucket_code2s})


@pytest.mark.risk_type('IRVega')
@pytest.mark.parametrize('scenario', ['low', 'medium', 'high'])
class TestIRVegaCalculator:
    def test_rho_correlation_matrices(self, calculator: IRVegaCalculator, scenario: str):
        rho = calculator.rho_correlation_matrices(scenario=scenario, bucket=None)

        assert rho is not None
        assert len(rho) == 1

        assert None in rho
        assert rho[None].shape == (35, 35)
