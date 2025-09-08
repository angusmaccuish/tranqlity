import pandas as pd
import pytest

from tranql.models.frtb.sa.calculator.drc_calculator import DRCNonSecCalculator


@pytest.fixture
def calculator(calculator_factory):
    return calculator_factory(DRCNonSecCalculator)


@pytest.mark.risk_type('JTD')
class TestDRCNonSecCalculator:
    def test_calculate_levels(self, calculator: DRCNonSecCalculator, dataframe_builder):
        sensis = dataframe_builder(
            '+---------+------------+-----------+--------------+----------------+-----------+----------------+',
            '| GroupID | FRTBBucket | ObligorId | RWLabel      | DebtSeniority  | ShareType | ScaledGrossJTD |',
            '+---------+------------+-----------+--------------+----------------+-----------+----------------+',
            '|       1 | corporates | ObligorA  | corporates_A | EQUITY         | NOTFUND   |           50.0 |',
            '|       1 | corporates | ObligorA  | corporates_A | SENIOR SECURED | NOTFUND   |          100.0 |',
            '+---------+------------+-----------+--------------+----------------+-----------+----------------+')

        result = calculator.calculate_levels(sensis=sensis)

        assert result is not None
        assert len(result) == 2
        assert 'total' in result
        assert 'bucket' in result
