import pytest

from tranql.models.frtb.sa.calculator.curvature_calculators import FXCurvatureCalculator


@pytest.mark.risk_type('FXGamma')
class TestFXCurvatureCalculator:
    def test_no_other_bucket(self, calculator_factory):
        calculator = calculator_factory(FXCurvatureCalculator)
        assert calculator.other_bucket is None
