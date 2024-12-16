import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tranql.models.frtb.sa.calculator.rrao_calculator import RRAOCalculator


@pytest.fixture
def calculator(calculator_factory):
    return calculator_factory(RRAOCalculator, correlations=None)


# noinspection SpellCheckingInspection
@pytest.mark.risk_type('RRAO')
class TestRRAOCalculator:
    def test_calculate_levels(self, calculator: RRAOCalculator):
        sensis = pd.DataFrame({
            'GroupID': [1, 1, 2],
            'RWLabel': ['NA_RRAO_EXOTIC_NOT-LISTED', 'NA_RRAO_EXOTIC_NOT-LISTED', 'NA_RRAO_OTHER_NOT-LISTED'],
            'AbsoluteRiskValue': [1000.0, 2000.0, 5000.0],
        })

        result = calculator.calculate_levels(sensis=sensis)

        assert result is not None
        assert 'total' in result

        expected = pd.DataFrame({
            'GroupID': [1, 2],
            'RRAO.Notional': [3000.0, 5000.0],
            'RRAO': [30.0, 5.0],
        })

        assert_frame_equal(left=expected.set_index('GroupID'),
                           right=result['total'],
                           check_like=True,
                           check_names=False)
