import pytest

from tranql.models.frtb.sa.calculator.curvature_calculators import CMCurvatureCalculator


@pytest.mark.risk_type('CMGamma')
class TestCMCurvatureCalculator:
    def test_no_other_bucket(self, calculator_factory):
        calculator = calculator_factory(CMCurvatureCalculator)
        assert calculator.other_bucket is None
