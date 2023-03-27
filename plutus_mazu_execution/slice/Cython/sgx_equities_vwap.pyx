import math
import numpy as np
import pandas as pd


class SGXEquitiesVwap:
    def __init__(self, strategy_config):
        # data structure propagated from execution model
        self.strategy_config = strategy_config

        # internal data structures
        self.mean_weight_of_30s = {}
        self.mean_weight_of_5min = {}
        self.volume_weight = {}
        self.child_orders = {}

        self.equities_and_30s_volume_data = {}
        self.equities_and_5min_volume_data = {}

    def collect_volume_data(self, equities_and_short_interval_dfs, equities_and_long_interval_dfs, updating=False):
        """Collect the volumes from the short and long interval dataframes containing the last size and clean it
        """
        for equity in equities_and_short_interval_dfs.keys():
            if equities_and_short_interval_dfs[equity] is not None:
                short_interval_volume_series = equities_and_short_interval_dfs[equity]['last_size'].fillna(0).astype(np.uint64).rename(equity)
                long_interval_volume_series = equities_and_long_interval_dfs[equity]['last_size'].fillna(0).astype(np.uint64).rename(equity)

                short_interval_volume_series = short_interval_volume_series / short_interval_volume_series.sum()
                long_interval_volume_series = long_interval_volume_series / long_interval_volume_series.sum()

                self.equities_and_30s_volume_data[equity].append(short_interval_volume_series)
                self.equities_and_5min_volume_data[equity].append(long_interval_volume_series)

        # When updating is set to True, pop oldest data point
        if updating:
            for equity in equities_and_short_interval_dfs.keys():
                if equities_and_short_interval_dfs[equity] is not None:
                    self.equities_and_30s_volume_data[equity].pop(0)
                    self.equities_and_5min_volume_data[equity].pop(0)

    def process_volume_data(self):
        """
        Processes the volume data collected to calculate mean weights of 30 second intervals and 5 min intervals
        """
        short_interval_df = pd.concat([pd.concat(list_of_series).rename(equity) for equity, list_of_series in
                                       self.equities_and_30s_volume_data.items()], axis=1)
        long_interval_df = pd.concat([pd.concat(list_of_series).rename(equity) for equity, list_of_series in
                                      self.equities_and_5min_volume_data.items()], axis=1)
        self.mean_weight_of_30s = short_interval_df.groupby(lambda dt: dt.time).sum() / len(set(short_interval_df.index.date))
        self.mean_weight_of_5min = long_interval_df.groupby(lambda dt: dt.time).sum() / len(set(long_interval_df.index.date))

    def generate_weight(self):
        """
        Calculate weight from historical data for the equity, used for slicing order vector.
        """
        # Reindex the 5min df with the 30s df and ffill (equivalent of a merge)
        reindexed_mean_weight_long_interval_df = self.mean_weight_of_5min.reindex(self.mean_weight_of_30s.index).ffill()
        volume_weight = self.mean_weight_of_30s / reindexed_mean_weight_long_interval_df
        self.volume_weight = volume_weight.replace(np.inf, np.nan).fillna(0)

    def slice_order(self, order_vector, time_now):
        """ Generate child orders based on weights
        """
        child_order = {}
        record_order = {}
        order_weight = {}
        for equity in order_vector:
            record_order[equity] = 0
            order_weight[equity] = [self.volume_weight[equity][self.volume_weight[equity].index.searchsorted(
                (time_now + i * pd.Timedelta('30s')).time()) - 1] for i in range(10)]
        for i in range(9):
            for equity, order in order_vector.items():
                if order >= 0:
                    child_order[equity] = min(math.ceil(order_weight[equity][i] / np.sum(order_weight[equity]) * order), order - record_order[equity])
                else:
                    child_order[equity] = max(math.floor(order_weight[equity][i] / np.sum(order_weight[equity]) * order), order - record_order[equity])

                record_order[equity] += child_order[equity]

            self.child_orders[i] = child_order.copy()
        for equity, order in order_vector.items():
            child_order[equity] = order - record_order[equity]
        self.child_orders[9] = child_order.copy()

        return self.child_orders

    def initialise_data_structures(self, list_of_equities):
        """Assign universe to data structures
        """
        self.equities_and_30s_volume_data = {equity: [] for equity in list_of_equities}
        self.equities_and_5min_volume_data = {equity: [] for equity in list_of_equities}
        self.mean_weight_of_30s = {equity: None for equity in list_of_equities}
        self.mean_weight_of_5min = {equity: None for equity in list_of_equities}
        self.volume_weight = {equity: None for equity in list_of_equities}
