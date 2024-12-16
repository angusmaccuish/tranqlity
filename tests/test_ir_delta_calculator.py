import pytest

from tranql.models.frtb.sa.calculator.delta_calculators import IRDeltaCalculator


@pytest.fixture
def calculator(calculator_factory):
    bucket_codes = ['0.25Y', '0.5Y', '1Y', '2Y', '3Y', '5Y', '10Y', '15Y', '20Y', '30Y']
    return calculator_factory(IRDeltaCalculator, params={'BucketCode': bucket_codes})


@pytest.mark.risk_type('IRDelta')
@pytest.mark.parametrize('scenario', ['low', 'medium', 'high'])
class TestIRDeltaCalculator:
    def test_rho_correlation_matrices(self, calculator: IRDeltaCalculator, scenario: str):
        rho = calculator.rho_correlation_matrices(scenario=scenario, bucket=None)

        assert rho is not None
        assert len(rho) == 2

        assert 'Underlying' in rho
        assert rho['Underlying'].shape == (12, 12)

        assert None in rho
        assert rho[None].shape == (12, 12)
