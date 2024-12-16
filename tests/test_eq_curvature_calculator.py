import pytest

from tranql.models.frtb.sa.calculator.curvature_calculators import EQCurvatureCalculator


@pytest.mark.risk_type('EQGamma')
class TestEQCurvatureCalculator:
    def test_default_other_bucket(self, calculator_factory):
        calculator = calculator_factory(EQCurvatureCalculator)
        assert calculator.other_bucket == '11'

    def test_other_bucket_override(self, calculator_factory):
        calculator = calculator_factory(EQCurvatureCalculator, params={'sa.equity.otherSector': 12})
        assert calculator.other_bucket == '12'
