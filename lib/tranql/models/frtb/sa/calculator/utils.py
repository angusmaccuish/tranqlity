from functools import wraps
from typing import Dict

import numpy as np
import pandas as pd


def explode_bucket_code(fn):
    @wraps(fn)
    def inner(self, sensis: pd.DataFrame, *args, **kwargs) -> Dict[str, pd.DataFrame]:
        bucket_code_aggregation = self.params.get('sa.aggregation.bucketcode', 'N') == 'Y'
        if bucket_code_aggregation and 'BucketCode' in self.params:
            # We need to disaggregate the RiskValue array, one row per BucketCode (amend GroupID to keep separated)
            bucket_codes = self.params['BucketCode']
            count = len(bucket_codes)
            sensis['GroupID'] = sensis['GroupID'].map(lambda gid: [-gid*count-k for k in range(count)])
            sensis['BucketCode'] = [bucket_codes] * len(sensis)
            sensis['RiskValue'] = sensis['RiskValue'].map(lambda x: list(np.diag(x)))
            sensis = sensis.explode(['GroupID', 'BucketCode', 'RiskValue'])
            group_bucket_codes = sensis.set_index('GroupID')['BucketCode'].to_dict()

            result = fn(self, sensis, *args, **kwargs)

            for df in result.values():
                # Add the BucketCode associated with each disaggregated GroupID, then reinstate original GroupID's
                df['BucketCode'] = df['GroupID'].map(group_bucket_codes)
                df['GroupID'] = -df['GroupID'] // count

            return result
        else:
            # Compute as normal
            result = fn(self, sensis, *args, **kwargs)

            if bucket_code_aggregation:
                # BucketCode a requested dimension, but not applicable for this risk type (eg. RRAO) - default to N/A
                return {k: df.assign(BucketCode='N/A') for k, df in result.items()}
            else:
                return result

    return inner


def reindex_group_ids(fn):
    @wraps(fn)
    def inner(self, sensis: pd.DataFrame, *args, **kwargs) -> Dict[str, pd.DataFrame]:
        if 'GroupID' in sensis:
            reverse_mapping = sensis['GroupID'].drop_duplicates().reset_index(drop=True).to_dict()
            index = {v: k for k, v in reverse_mapping.items()}
            results = fn(self, sensis.assign(GroupID=sensis['GroupID'].map(index)), *args, **kwargs)
            return {k: df.assign(GroupID=df['GroupID'].map(reverse_mapping)) for k, df in results.items()}
        else:
            return fn(self, sensis, *args, **kwargs)

    return inner


def handle_risk_weight_measure(fn):
    @wraps(fn)
    def inner(self, sensis: pd.DataFrame, *args, **kwargs) -> Dict[str, pd.DataFrame]:
        result = fn(self, sensis, *args, **kwargs)

        if 'RiskWeightValue' in self.params['measure'] and 'RiskWeightValue' in sensis.columns and 'total' in result:
            # If the max risk weight == min risk weight per group then there exists a unique weight we can publish
            aggregates = {
                'MaxRiskWeightValue': lambda x: np.max(np.stack(x)),
                'MinRiskWeightValue': lambda x: np.min(np.stack(x))
            }
            df = sensis.groupby(by='GroupID', sort=False)['RiskWeightValue'].agg(**aggregates)

            unique_risk_weight = df['MinRiskWeightValue'] == df['MaxRiskWeightValue']

            if np.any(unique_risk_weight):
                df['RiskWeightValue'] = np.where(unique_risk_weight, df['MinRiskWeightValue'], np.nan)
                risk_weights = df['RiskWeightValue'].to_dict()
                total_df = result['total']
                total_df['RiskWeightValue'] = total_df['GroupID'].map(risk_weights)

        return result

    return inner
