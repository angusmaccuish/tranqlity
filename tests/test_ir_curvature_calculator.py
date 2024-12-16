import pytest

from tranql.models.frtb.sa.calculator.curvature_calculators import IRCurvatureCalculator


@pytest.mark.risk_type('IRGamma')
class TestIRCurvatureCalculator:
    def test_no_other_bucket(self, calculator_factory):
        calculator = calculator_factory(IRCurvatureCalculator)
        assert calculator.other_bucket is None
