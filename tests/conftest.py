from pathlib import Path

import pandas as pd
import pytest


def get_regulation(request):
    regulation_marker = request.node.get_closest_marker('regulation')
    return regulation_marker.args[0] if regulation_marker and regulation_marker.args else 'd457'


def get_risk_type(request):
    risk_type_marker = request.node.get_closest_marker('risk_type')
    assert risk_type_marker and risk_type_marker.args
    return risk_type_marker.args[0]


def load_csv(file):
    return pd.read_csv(Path(__file__).parent / 'data' / file)


@pytest.fixture
def calculator_factory(request, risk_weights, correlations, seniority_ranking):
    def create(cls, **kwargs):
        default_kwargs = dict(reg_name=get_regulation(request),
                              risk_weights=risk_weights,
                              correlations=correlations,
                              seniority_ranking=seniority_ranking,
                              params=None)

        return cls(**{**default_kwargs, **kwargs})

    return create


@pytest.fixture
def risk_weights(request):
    risk_type = get_risk_type(request)

    if risk_type == 'JTD':
        df = load_csv('drc_risk_weights.csv')
    else:
        df = load_csv('risk_weights.csv')
        df = df.loc[df['risk_type'] == risk_type]

    return df.loc[df['reg_name'] == get_regulation(request)]


@pytest.fixture
def correlations(request):
    risk_type = get_risk_type(request)

    if risk_type in ['JTD', 'RRAO']:
        return None

    df = load_csv('correlations.csv')
    df = df.loc[df['risk_type'] == risk_type]

    return df.loc[df['reg_name'] == get_regulation(request)]


@pytest.fixture
def seniority_ranking(request):
    risk_type = get_risk_type(request)

    if risk_type != 'JTD':
        return None

    return load_csv('seniority_ranking.csv')
