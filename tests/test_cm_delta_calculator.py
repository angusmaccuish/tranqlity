import pytest

from tranql.models.frtb.sa.calculator.delta_calculators import CMDeltaCalculator


@pytest.fixture
def calculator(calculator_factory):
    bucket_codes = ['0Y', '0.25Y', '0.5Y', '1Y', '2Y', '3Y', '5Y', '10Y', '15Y', '20Y', '30Y']
    return calculator_factory(CMDeltaCalculator, params={'BucketCode': bucket_codes})


@pytest.mark.risk_type('CMDelta')
@pytest.mark.parametrize('scenario', ['low', 'medium', 'high'])
@pytest.mark.parametrize('bucket', range(1, 12))
class TestCMDeltaCalculator:
    def test_rho_correlation_matrices(self, calculator: CMDeltaCalculator, scenario: str, bucket: str):
        rho = calculator.rho_correlation_matrices(scenario=scenario, bucket=str(bucket))

        assert rho is not None
        assert len(rho) == 4
        assert ('Underlying', 'CMTYLocation') in rho
        assert 'Underlying' in rho
        assert 'CMTYLocation' in rho
        assert None in rho

        for matrix in rho.values():
            assert matrix.shape == (11, 11)
