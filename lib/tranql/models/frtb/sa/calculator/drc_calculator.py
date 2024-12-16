from typing import Dict

import numpy as np
import pandas as pd

from .calculator import Calculator


class DRCNonSecCalculator(Calculator):
    def net_weighted_sensis(self, data: pd.DataFrame) -> pd.DataFrame:
        # Funds have special handling
        data['fund'] = np.where(data['ShareType'].isin(['FUND', 'ETF']), 'Y', 'N')

        # Look up risk weights using RWLabel
        data['RiskWeightValue'] = data['RWLabel'].map(self.risk_weights).fillna(0.0)

        # Net JTD, disaggregated by DebtSeniority
        agg_cols = ['fund', 'GroupID', 'FRTBBucket', 'ObligorId', 'RiskWeightValue', 'DebtSeniority']
        df = data.groupby(by=agg_cols, sort=False)['ScaledGrossJTD'].sum()

        # Unstack the DebtSeniority, map the resulting columns to their Seniority Rankings then sort
        seniority_ranking = self.seniority_ranking.set_index('DebtSeniority')['SeniorityRanking'].astype(int).to_dict()
        df = df.unstack('DebtSeniority', fill_value=0.0)
        missing = [col for col in df.columns if col not in seniority_ranking]
        assert not missing, f'The following have no seniority ranking: {missing}'
        df = df.rename(columns=seniority_ranking).sort_index(axis=1)

        # Sum across all Debt Seniority columns to obtain net ScaledGrossJTD
        scaled_jtd = df.sum(axis=1)

        # Initialise net_long_jtd to be first long ScaledGrossJTD of the first (most senior) column
        net_long_jtd = df.iloc[:, 0]
        net_long_jtd[net_long_jtd < 0] = 0.0

        # Netting - advance across the columns, in order of decreasing seniority
        for col in range(1, len(df.columns)):
            net_long_jtd += df.iloc[:, col]
            net_long_jtd[net_long_jtd < 0] = 0.0

        df = pd.DataFrame({'ScaledJTD': scaled_jtd, 'net_jtd_l': net_long_jtd})

        # No need to do the short netting version, long + short must = net so it can be derived
        df['net_jtd_s'] = df['ScaledJTD'] - df['net_jtd_l']

        # Apply risk weights
        risk_weights = df.index.get_level_values('RiskWeightValue')
        df['w_net_jtd_l'] = df['net_jtd_l'] * risk_weights
        df['w_net_jtd_s'] = df['net_jtd_s'] * risk_weights

        zero_risk_weights = risk_weights == 0.0

        if zero_risk_weights.any():
            # Check if we have to exclude zero-weighted jtd from the HBR calculation (by zero-ing the jtd)
            if self.params.get('sa.drc.ZeroRiskIncludeInHBR', 'Y') != 'Y':
                df.loc[zero_risk_weights, 'net_jtd_l'] = 0.0
                df.loc[zero_risk_weights, 'net_jtd_s'] = 0.0

        return df

    def calculate_levels(self, sensis: pd.DataFrame) -> Dict:
        net_weighted_sensis = self.net_weighted_sensis(sensis)
        bucket_df = net_weighted_sensis.groupby(level=['fund', 'GroupID', 'FRTBBucket']).sum(numeric_only=True)
        bucket_df['hbr'] = self._calculate_hedge_benefit_ratio(sensis=bucket_df)

        if 'Y' in bucket_df.index:
            # if fund risk is present, set the hedge benefit ratio to zero for these
            bucket_df.loc['Y', 'hbr'] = 0.0

        bucket_df['drc'] = self._calculate_default_risk_charge(sensis=bucket_df)
        bucket_df = bucket_df.groupby(level=['GroupID', 'FRTBBucket'], sort=False).sum().reset_index(level='FRTBBucket')
        total_df = self._calculate_totals(sensis=bucket_df)

        return {'total': total_df, 'bucket': bucket_df}

    @staticmethod
    def _calculate_hedge_benefit_ratio(sensis: pd.DataFrame):
        net_long_jtd, abs_net_short_jtd = sensis['net_jtd_l'], sensis['net_jtd_s'].abs()
        return np.where(net_long_jtd > 0, net_long_jtd / (net_long_jtd + abs_net_short_jtd), 0.0)

    @staticmethod
    def _calculate_default_risk_charge(sensis: pd.DataFrame):
        drc = sensis['w_net_jtd_l'] - sensis['hbr'] * sensis['w_net_jtd_s'].abs()
        return drc.clip(lower=0.0)

    @staticmethod
    def _calculate_totals(sensis: pd.DataFrame):
        total_measures = ['ScaledJTD', 'w_net_jtd_l', 'w_net_jtd_s', 'net_jtd_l', 'net_jtd_s', 'drc']
        totals = sensis.groupby(level='GroupID', sort=False)[total_measures].sum()

        # HBR should be a bucket-level measure, but add it here as it can be requested at the Obligor/Tranche levels
        return totals.assign(hbr=DRCNonSecCalculator._calculate_hedge_benefit_ratio(sensis=totals))


class DRCSecCalculator(DRCNonSecCalculator):
    def net_weighted_sensis(self, data: pd.DataFrame) -> pd.DataFrame:
        agg_cols = ['GroupID', 'FRTBBucket', 'FRTBDRCTranche', 'DRCSecRiskWeight']
        df = data.groupby(by=agg_cols, as_index=False, sort=False).agg(ScaledJTD=('ScaledGrossJTD', 'sum'))
        df['net_jtd_l'] = df['ScaledJTD'].clip(lower=0.0)
        df['net_jtd_s'] = df['ScaledJTD'].clip(upper=0.0)

        risk_weights = df['DRCSecRiskWeight'].astype(np.float64)
        df['w_net_jtd_l'] = df['net_jtd_l'] * risk_weights
        df['w_net_jtd_s'] = df['net_jtd_s'] * risk_weights

        return df

    def calculate_levels(self, sensis: pd.DataFrame) -> Dict:
        net_weighted_sensis = self.net_weighted_sensis(sensis)
        bucket_df = net_weighted_sensis.groupby(by=['GroupID', 'FRTBBucket'], sort=False).sum(numeric_only=True)
        bucket_df['hbr'] = self._calculate_hedge_benefit_ratio(sensis=bucket_df)
        bucket_df['drc'] = self._calculate_default_risk_charge(sensis=bucket_df)
        bucket_df = bucket_df.reset_index(level='FRTBBucket')

        total_df = self._calculate_totals(sensis=bucket_df)

        return {'total': total_df, 'bucket': bucket_df}
