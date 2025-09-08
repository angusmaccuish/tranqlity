import io
import re
from pathlib import Path
from typing import Callable, Optional, List, Tuple

import numpy as np
import pandas as pd
import pytest


def get_regulation(request) -> str:
    regulation_marker = request.node.get_closest_marker('regulation')
    return regulation_marker.args[0] if regulation_marker and regulation_marker.args else 'd457'


def get_risk_type(request) -> str:
    risk_type_marker = request.node.get_closest_marker('risk_type')
    assert risk_type_marker and risk_type_marker.args
    return risk_type_marker.args[0]


def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(Path(__file__).parent / 'data' / file)


@pytest.fixture
def calculator_factory(request, risk_weights, correlations, seniority_ranking) -> Callable:
    def create(cls, **kwargs):
        default_kwargs = dict(reg_name=get_regulation(request),
                              risk_weights=risk_weights,
                              correlations=correlations,
                              seniority_ranking=seniority_ranking,
                              params=None)

        return cls(**{**default_kwargs, **kwargs})

    return create


@pytest.fixture
def risk_weights(request) -> pd.DataFrame:
    risk_type = get_risk_type(request)

    if risk_type == 'JTD':
        df = load_csv('drc_risk_weights.csv')
    else:
        df = load_csv('risk_weights.csv')
        df = df.loc[df['risk_type'] == risk_type]

    return df.loc[df['reg_name'] == get_regulation(request)]


@pytest.fixture
def correlations(request) -> Optional[pd.DataFrame]:
    risk_type = get_risk_type(request)

    if risk_type in ['JTD', 'RRAO']:
        return None

    df = load_csv('correlations.csv')
    df = df.loc[df['risk_type'] == risk_type]

    return df.loc[df['reg_name'] == get_regulation(request)]


@pytest.fixture
def seniority_ranking(request) -> Optional[pd.DataFrame]:
    risk_type = get_risk_type(request)

    if risk_type != 'JTD':
        return None

    return load_csv('seniority_ranking.csv')


class Table:
    @staticmethod
    def to_dataframe(*lines, index=None, line_terminator=';', **kwargs) -> pd.DataFrame:
        """Convert a table into a DataFrame."""
        # Remove whitespace before and after column separator and at start/end of line
        ignore_regex = re.compile(r'^\+(-+\+)+$')
        sub_regex = re.compile(r'\s*\|\s*')
        rows = [sub_regex.sub('|', line.strip()) for line in lines if not ignore_regex.match(line)]

        # Read csv into DataFrame
        df = pd.read_csv(io.StringIO(line_terminator.join(rows)), sep='|', lineterminator=line_terminator, **kwargs)

        # | Col1 | Col2 | formatting introduces empty columns, remove them
        drop_cols = [col for col in df.columns if col.startswith('Unnamed') and df[col].isna().all()]

        if drop_cols:
            df = df.drop(columns=drop_cols)

        # Array columns specified using [] brackets, for example [Shocks]
        array_cols = {col: col[1:-1] for col in df.columns if re.match(r'\[.*]', col)}

        if array_cols:
            # Special handling for array columns - convert to numpy arrays and drop the [] outer brackets from col name
            for col in array_cols:
                df[col] = df[col].map(eval).map(np.array)
            df = df.rename(columns=array_cols)

        # Create and sort index if specified
        return df if index is None else df.set_index(keys=index).sort_index()

    @staticmethod
    def to_series(*lines, line_terminator=';', **kwargs) -> pd.Series:
        """Convert a table into a Series."""
        df = Table.to_dataframe(*lines, line_terminator=line_terminator, **kwargs)

        if len(df.columns) > 1:
            # All columns bar the last are keys for the Series index
            df = df.set_index(keys=df.columns[:-1].tolist())

        return df.iloc[:, 0]


@pytest.fixture
def table() -> Table:
    """Return table instance so that tests can call to_* methods to create pandas objects."""
    return Table()
