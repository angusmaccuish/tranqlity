import pytest

from tranql.models.frtb.sa.calculator.curvature_calculators import CSRCurvatureCalculator


@pytest.mark.regulation('crr2')
@pytest.mark.risk_type('CSRGamma')
class TestCSRCurvatureCalculator:
    def test_default_other_bucket(self, calculator_factory):
        calculator = calculator_factory(CSRCurvatureCalculator)
        assert calculator.other_bucket == '18'

    def test_other_bucket_override(self, calculator_factory):
        calculator = calculator_factory(CSRCurvatureCalculator, params={'sa.csrnonsec.otherSector': 19})
        assert calculator.other_bucket == '19'
