import pytest

from tranql.models.frtb.sa.calculator.curvature_calculators import CSRSecCurvatureCalculator


@pytest.mark.risk_type('CSRSecGamma')
class TestCSRSecCurvatureCalculator:
    def test_default_other_bucket(self, calculator_factory):
        calculator = calculator_factory(CSRSecCurvatureCalculator)
        assert calculator.other_bucket == '25'

    def test_other_bucket_override(self, calculator_factory):
        calculator = calculator_factory(CSRSecCurvatureCalculator, params={'sa.csrsec.otherSector': 26})
        assert calculator.other_bucket == '26'
