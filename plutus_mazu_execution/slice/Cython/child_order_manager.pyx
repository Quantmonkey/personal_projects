import warnings
import numpy as np
import pandas as pd

from helper_models.general.exception_handler import ExceptionHandler
warnings.filterwarnings('error', category=RuntimeWarning)
exception_handler = ExceptionHandler()


class ChildOrderManager:
    def __init__(self, strategy_config):
        self.strategy_config = strategy_config
        self.verbose = self.strategy_config.verbose

        # internal data structures
        self.trading_hours = []
        self.order_vector = {}
        self.equities_and_urgency = {}  # urgency determines aggressiveness of order
        self.equities_and_ticks_since_last_order = {}
        self.equities_and_maximum_waiting_ticks = {}
        self.equities_and_current_limit_orders = {}
        self.equities_and_order_status = {}
        self.wide_spread_threshold = None

        # data structures propagated from universe model
        self.equities_and_id_for_trading = {}
        self.equities_and_tick_size = {}

        # data structure propagated from data model
        self.equities_and_latest_tick = {}

        # data structures for dynamic updating
        self.equities_and_mean_ask_volume = {}
        self.equities_and_mean_bid_volume = {}
        self.equities_and_ticks_in_each_period = {}
        self.equities_and_ticks_in_each_period_historical_list = {}
        self.equities_and_mid_price_volatility = {}
        self.equities_and_mean_price_spread = {}
        self.equities_and_mean_price_spread_historical_list = {}
        self.equities_and_mean_bid_volume_historical_list = {}
        self.equities_and_mean_ask_volume_historical_list = {}

        # lock to prevent sending more orders until an order status is received
        self.waiting_for_order_status = False

    def get_all_benchmarks(self, equity):
        """
        Calculates benchmarks for spread size, volume, volatility, tick frequency, and waiting time for each equity,
        based on its historical data.
        """
        # Calculate the mean price spread for the equity
        self.get_mean_price_spread(equity)

        # Calculate mean bid and ask volumes for distinguishing between genuine traders and market manipulators
        self.get_mean_volumes(equity)

        # Get average number of ticks coming in each child order period
        self.get_mean_ticks_in_each_period(equity)

        # Calculate maximum waiting ticks for each equity
        self.get_maximum_waiting_ticks(equity)

    @exception_handler.except_get_wide_spread_threshold
    def get_wide_spread_threshold(self):
        """
        Calculates the threshold for an equity's spread to be considered wide, based on a quantiling of historical mean
        spreads across all equities.
        """
        equities_and_spread_in_number_of_ticks = []

        for equity, price_spread in self.equities_and_mean_price_spread.items():
            tick_size = float(self.equities_and_tick_size[equity])
            spread_in_number_of_ticks = price_spread // tick_size
            equities_and_spread_in_number_of_ticks.append(spread_in_number_of_ticks)

        threshold = pd.Series(equities_and_spread_in_number_of_ticks).quantile(
            self.strategy_config.WIDE_PRICE_SPREAD_PERCENTILE)

        self.wide_spread_threshold = threshold

    def get_mean_volumes(self, equity):
        """
        Calculates mean bid and ask volume for this equity. Volumes need to be grouped by each trading hour
        since volume traded is not uniform throughout the day.
        """

        # For each trading hour
        for hour in self.trading_hours:
            total_data_points_for_this_hour_in_historical_time_frame = \
                sum(map(lambda pair: pair[1], self.equities_and_mean_bid_volume_historical_list[equity][hour]))

            total_bid_volume_for_this_hour_in_historical_time_frame = \
                sum(map(lambda pair: pair[0], self.equities_and_mean_bid_volume_historical_list[equity][hour]))

            total_ask_volume_for_this_hour_in_historical_time_frame = \
                sum(map(lambda pair: pair[0], self.equities_and_mean_ask_volume_historical_list[equity][hour]))

            mean_bid_volume = total_bid_volume_for_this_hour_in_historical_time_frame / total_data_points_for_this_hour_in_historical_time_frame
            mean_ask_volume = total_ask_volume_for_this_hour_in_historical_time_frame / total_data_points_for_this_hour_in_historical_time_frame

            self.equities_and_mean_bid_volume[equity][hour] = mean_bid_volume
            self.equities_and_mean_ask_volume[equity][hour] = mean_ask_volume

    def get_mean_ticks_in_each_period(self, equity):
        """
        Calculate the mean number of ticks coming in for this equity in each period. This period needs to be
        grouped by trading hour since ticks coming in is not uniform throughout the day
        """
        # For each trading hour
        for hour in self.trading_hours:
            self.equities_and_ticks_in_each_period[equity][hour] = \
                sum(self.equities_and_ticks_in_each_period_historical_list[equity][hour]) // \
                len(self.equities_and_ticks_in_each_period_historical_list[equity][hour])

    def get_mean_price_spread(self, equity):
        """Calculate the average spread of ticks for this equity
        """
        total_data_points_for_this_hour_in_historical_time_frame = \
            sum(map(lambda pair: pair[1], self.equities_and_mean_price_spread_historical_list[equity]))

        total_price_spread_for_this_hour_in_historical_time_frame = \
            sum(map(lambda pair: pair[0], self.equities_and_mean_price_spread_historical_list[equity]))

        mean_price_spread = total_price_spread_for_this_hour_in_historical_time_frame / total_data_points_for_this_hour_in_historical_time_frame

        self.equities_and_mean_price_spread[equity] = mean_price_spread

    def get_maximum_waiting_ticks(self, equity):
        """
        Calculates the maximum waiting ticks for each equity. The calculation is based on comparing equity's
        volatility with its past 30 days mean volatility, and multiplying the fraction with mean number of ticks
        in 30 seconds. Calculations should be grouped by trading hour.
        """

        # For each trading hour
        for hour in self.trading_hours:

            average_ticks = self.equities_and_ticks_in_each_period[equity][hour]
            historical_volatility = self.equities_and_mid_price_volatility[equity][hour]
            latest_volatility = historical_volatility[-1]

            # If we don't have any data in the latest day, assume that volatility is same as mean
            if np.isnan(latest_volatility):
                multiplier = 1
            # If historical volatility is very small, set multiplier to 1 (avoid division by zero)
            elif np.nanmean(historical_volatility) < 1e-6:
                multiplier = 1
            else:
                multiplier = min(latest_volatility / np.nanmean(historical_volatility), 1)

            waiting_ticks = np.ceil(average_ticks * multiplier)
            self.equities_and_maximum_waiting_ticks[equity][hour] = waiting_ticks

    def assert_wide_spread(self, tick, equity):
        """ Judge whether the equity's spread is wide or narrow. Return True if wide, False if narrow.
        """
        tick_size = float(self.equities_and_tick_size[equity])
        spread = float(tick.ask - tick.bid) / tick_size
        return spread > self.wide_spread_threshold

    def get_micro_price(self, hour, equity, tick):
        """
        Calculate micro price for the equity. Sets suspiciously low volumes on either bid or ask
        sides to the mean volume for our calculations, in order to combat market manipulators.
        """

        vol_bid = max(int(tick.vol_bid), self.equities_and_mean_bid_volume[equity][hour])
        vol_ask = max(int(tick.vol_ask), self.equities_and_mean_ask_volume[equity][hour])

        micro_price = (vol_bid * float(tick.ask) + vol_ask * float(tick.bid)) / (vol_bid + vol_ask)
        return micro_price

    @exception_handler.except_record_ticks_and_check_if_need_to_send_order
    def record_ticks_and_check_if_need_to_send_order(self, latest_tick, time_now):
        """ Called each tick to update waiting ticks for each equity.
        """
        # Check current trading hour to find out what is maximum waiting ticks
        hour = time_now.hour
        for equity, id in self.equities_and_id_for_trading.items():

            # Check latest tick belongs to which security
            if latest_tick.security_id == id:
                self.equities_and_ticks_since_last_order[equity] += 1

                # If we have waited the maximum number of ticks, send next order and reset.
                if self.equities_and_ticks_since_last_order[equity] >= self.equities_and_maximum_waiting_ticks[equity][hour] and \
                        not self.waiting_for_order_status:
                    self.equities_and_ticks_since_last_order[equity] = 0
                    order = self.create_order(latest_tick, equity, time_now)
                    return order

        # This means that we are not yet ready to send an order
        return {"qty": 0}

    def create_order(self, latest_tick, equity, time_now):
        """ Creates and order to pass to execution model. This function is called whenever maximum waiting ticks is up.
        """
        # Check if there are unfilled quantities
        if equity in self.equities_and_order_status and self.equities_and_order_status[equity].status != 'CANCELED':
            order_has_quantities_unfilled = int(self.equities_and_order_status[equity].remaining_quantity) != 0
        else:
            order_has_quantities_unfilled = False

        # If there are unfulfilled orders, cancel them and add urgency.
        if order_has_quantities_unfilled:
            # Canceling orders may fail in the (very rare) event that the order is
            # filled when we are still halfway in the main event loop
            try:
                self.strategy_config.ALGOTRADER_CANCEL_ORDER_FUNCTION(self.equities_and_order_status[equity].int_id)
                side = 1 if 'BUY' in self.equities_and_order_status[equity].int_id else -1
                self.order_vector[equity] += side * int(self.equities_and_order_status[equity].remaining_quantity)
                del self.equities_and_order_status[equity]
                self.equities_and_urgency[equity] += 1
            except:
                self.equities_and_urgency[equity] = 0

        # Child order is completed, reset urgency
        else:
            self.equities_and_urgency[equity] = 0

        # Check whether spread is wide or narrow to determine order type
        wide_spread = self.assert_wide_spread(latest_tick, equity)
        if wide_spread:
            order = self.create_wide_spread_order(latest_tick, equity, time_now)
        else:
            order = self.create_narrow_spread_order(latest_tick, equity, time_now)

        return order

    def create_wide_spread_order(self, tick, equity, time_now):
        """
        Creates an order for an equity with a wide price spread. A critical value is calculated based on micro price,
        and a couple of dice rolls are made based on urgency. This critical value is compared with dice rolls outcome to
        determine if we should be aggressive or not.

        # TODO TEMP CHANGE Aggressive: Send FOK for full child order, followed by a random sized MLO, at best quote.
        Passive: Send random sized LO, at micro price.
        """
        # Number of dice rolls to make determined by urgency
        dice_rolls = np.random.uniform(size=self.equities_and_urgency[equity] + 1)
        required_quantity = self.order_vector[equity]
        trading_hour = time_now.hour
        tick_size = float(self.equities_and_tick_size[equity])

        # Calculate the micro price, which will be used in determining aggressiveness
        micro_price = self.get_micro_price(trading_hour, equity, tick)

        if required_quantity > 0:
            critical_value = (1 - (float(tick.ask) - micro_price) / float(
                tick.ask - tick.bid)) ** self.strategy_config.AGGRESSIVE_EXPONENT
            if dice_rolls.min() < critical_value:
                order_quantity = min(abs(required_quantity), float(tick.vol_ask))
                limit_price = float(tick.ask)
            else:
                order_quantity = np.random.randint(1, abs(required_quantity) + 1)
                limit_price = round(micro_price / tick_size) * tick_size

        elif required_quantity < 0:
            critical_value = (1 - (micro_price - float(tick.bid)) / float(
                tick.ask - tick.bid)) ** self.strategy_config.AGGRESSIVE_EXPONENT
            # Go aggressive
            if dice_rolls.min() < critical_value:
                order_quantity = min(abs(required_quantity), float(tick.vol_bid))
                limit_price = float(tick.bid)
            else:
                order_quantity = np.random.randint(1, abs(required_quantity) + 1)
                limit_price = round(micro_price / tick_size) * tick_size
        # No required quantity. Don't need to create an order
        else:
            return {"qty": 0}
        side = 'BUY' if required_quantity > 0 else 'SELL'

        return {"equity": equity, "side": side, "qty": order_quantity, "tif": "GTC",
                "limit_price": limit_price}

    def create_narrow_spread_order(self, tick, equity, time_now):
        """
        Creates an order for an equity with a narrow price spread. A critical value is calculated based on
        order book imbalance,and a couple of dice rolls are made based on urgency. This critical value is compared
        with dice rolls outcome to determine if we should be aggressive or not.

        # TODO TEMP CHANGE Aggressive: Send FOK for full child order, followed by a random sized MLO, at best quote.
        Passive: Send random sized LO, at opposite side of best quote.
        """

        dice_rolls = np.random.uniform(size=self.equities_and_urgency[equity] + 1)
        required_quantity = self.order_vector[equity]
        trading_hour = time_now.hour

        # Calculate the micro price, which will be used in determining aggressiveness
        micro_price = self.get_micro_price(trading_hour, equity, tick)

        if required_quantity > 0:
            critical_value = (1 - (float(tick.ask) - micro_price) / float(
                tick.ask - tick.bid)) ** self.strategy_config.AGGRESSIVE_EXPONENT
            if dice_rolls.min() < critical_value:
                order_quantity = min(abs(required_quantity), float(tick.vol_ask))
                limit_price = float(tick.ask)
            else:
                order_quantity = np.random.randint(1, abs(required_quantity) + 1)
                limit_price = float(tick.bid)

        elif required_quantity < 0:
            critical_value = (1 - (micro_price - float(tick.bid)) / float(
                tick.ask - tick.bid)) ** self.strategy_config.AGGRESSIVE_EXPONENT
            if dice_rolls.min() < critical_value:
                order_quantity = min(abs(required_quantity), float(tick.vol_bid))
                limit_price = float(tick.bid)
            else:
                order_quantity = np.random.randint(1, abs(required_quantity) + 1)
                limit_price = float(tick.ask)

        # No required quantity. Don't need to create an order
        else:
            return {"qty": 0}
        side = 'BUY' if required_quantity > 0 else 'SELL'

        return {"equity": equity, "side": side, "qty": order_quantity, "tif": "GTC",
                "limit_price": limit_price}

    def initialise_data_structures(self):
        """ Initiates all data structures needed for the Child Order Manager
        """
        # Update trading hours for child order manager
        self.trading_hours.clear()
        for interval in self.strategy_config.TRADING_HOURS['trading_intervals']:
            self.trading_hours.extend(list(range(interval[0].hour, interval[1].hour)))

        # Assign universe to data structures
        for equity in self.equities_and_id_for_trading:
            self.equities_and_ticks_since_last_order[equity] = 0
            self.equities_and_urgency[equity] = 0
            self.equities_and_mean_price_spread[equity] = None
            self.equities_and_mean_price_spread_historical_list[equity] = []
            self.equities_and_mean_bid_volume[equity] = {hour: None for hour in self.trading_hours}
            self.equities_and_mean_ask_volume[equity] = {hour: None for hour in self.trading_hours}
            self.equities_and_ticks_in_each_period[equity] = {hour: None for hour in self.trading_hours}
            self.equities_and_maximum_waiting_ticks[equity] = {hour: None for hour in self.trading_hours}
            self.equities_and_mid_price_volatility[equity] = {hour: [] for hour in self.trading_hours}
            self.equities_and_mean_bid_volume_historical_list[equity] = {hour: [] for hour in self.trading_hours}
            self.equities_and_mean_ask_volume_historical_list[equity] = {hour: [] for hour in self.trading_hours}
            self.equities_and_ticks_in_each_period_historical_list[equity] = {hour: [] for hour in self.trading_hours}

    def receive_data_structures_from_execution_model(self, data_structures):
        """TODO
        """
        self.equities_and_id_for_trading = data_structures['trading universe']
        self.equities_and_tick_size = data_structures['equities and tick size']
        self.equities_and_current_limit_orders = data_structures['equities and limit orders']
        self.equities_and_order_status = data_structures['equities and order status']
        self.order_vector = data_structures['order vector']

    def collect_and_process_tick_data(self, counts_per_completion_period_dfs,
                                      volumes_and_counts_dfs, mid_prices_and_price_spreads_dfs, updating=False):
        """Vectorized collecting and updating all statistics needed for child order manager to function per day.
        Then iterate to convert the dataframes used in vectorized operations to load into the dynamic data structures (lists and dicts)
        """
        # Filter out any securities with None instead of df, or is redundantly filled with Nones
        securities_with_none_count_dfs = {security for security, df in counts_per_completion_period_dfs.items() if df is None}
        securities_with_none_volume_dfs = {security for security, df in volumes_and_counts_dfs.items() if df is None}
        securities_with_none_prices_dfs = {security for security, df in mid_prices_and_price_spreads_dfs.items() if df is None}
        securities_with_none_dfs = securities_with_none_count_dfs.union(securities_with_none_volume_dfs).union(securities_with_none_prices_dfs)

        # Remove them from all the dictionaries
        for security in securities_with_none_dfs:
            counts_per_completion_period_dfs.pop(security)
            volumes_and_counts_dfs.pop(security)
            mid_prices_and_price_spreads_dfs.pop(security)

        # If there is nothing left, just skip
        if len(counts_per_completion_period_dfs) == 0:
            return

        # counts_per_completion_period
        mean_ticks_per_completion_period = pd.concat({security: df['count'] for security, df in counts_per_completion_period_dfs.items()}, axis=1)
        mean_ticks_per_completion_period = mean_ticks_per_completion_period.groupby(lambda dt: dt.hour).mean()

        # volumes_and_counts
        multi_level_volumes_and_counts_df = pd.concat(volumes_and_counts_dfs, axis=1).fillna(0).astype(np.uint64)
        grouped_volumes_and_counts_df = multi_level_volumes_and_counts_df.groupby(lambda dt: dt.hour).sum()
        total_vol_bid_for_each_hour = grouped_volumes_and_counts_df.xs('vol_bid', axis=1, level=1)
        total_vol_ask_for_each_hour = grouped_volumes_and_counts_df.xs('vol_ask', axis=1, level=1)
        total_tick_counts_for_each_hour = grouped_volumes_and_counts_df.xs('count', axis=1, level=1)
        total_tick_counts_for_all_hours = total_tick_counts_for_each_hour.sum()

        # mid_prices_and_price_spreads
        multi_level_mid_prices_and_price_spreads_df = pd.concat(mid_prices_and_price_spreads_dfs, axis=1).astype(np.float64)
        multi_level_mid_prices_and_price_spreads_df.index = multi_level_mid_prices_and_price_spreads_df.index.hour
        mid_prices_std_for_each_hour = multi_level_mid_prices_and_price_spreads_df.xs('mid_price_std', axis=1, level=1)
        price_spreads_for_each_hour = multi_level_mid_prices_and_price_spreads_df.xs('price_spread', axis=1, level=1)
        price_spread_sums = price_spreads_for_each_hour.sum(axis=0)

        # Iterate to convert dataframes into list collections (transition from vectorized to dynamic operations)
        for equity in (self.equities_and_id_for_trading.keys() - securities_with_none_dfs):
            self.equities_and_mean_price_spread_historical_list[equity].append((price_spread_sums[equity], total_tick_counts_for_all_hours[equity]))

            for hour in self.trading_hours:
                ticks_in_this_hour = total_tick_counts_for_each_hour[equity][hour]

                if ticks_in_this_hour > 0:
                    total_bid_volume_in_this_hour = total_vol_bid_for_each_hour[equity][hour]
                    total_ask_volume_in_this_hour = total_vol_ask_for_each_hour[equity][hour]
                    mid_price_volatility_in_this_hour = mid_prices_std_for_each_hour[equity][hour]
                    mean_ticks_per_completion_period_in_this_hour = mean_ticks_per_completion_period[equity][hour]

                    self.equities_and_mean_bid_volume_historical_list[equity][hour].append((total_bid_volume_in_this_hour, ticks_in_this_hour))
                    self.equities_and_mean_ask_volume_historical_list[equity][hour].append((total_ask_volume_in_this_hour, ticks_in_this_hour))
                    self.equities_and_mid_price_volatility[equity][hour].append(mid_price_volatility_in_this_hour)
                    self.equities_and_ticks_in_each_period_historical_list[equity][hour].append(mean_ticks_per_completion_period_in_this_hour)

                else:
                    self.equities_and_mean_bid_volume_historical_list[equity][hour].append((0, 0))
                    self.equities_and_mean_ask_volume_historical_list[equity][hour].append((0, 0))
                    self.equities_and_mid_price_volatility[equity][hour].append(np.nan)
                    self.equities_and_ticks_in_each_period_historical_list[equity][hour].append(0)

        # When updating is set to True, pop oldest data point
        if updating:
            for equity in (self.equities_and_id_for_trading.keys() - securities_with_none_dfs):
                self.equities_and_mean_price_spread_historical_list[equity].pop(0)

                for hour in self.trading_hours:
                    self.equities_and_mean_bid_volume_historical_list[equity][hour].pop(0)
                    self.equities_and_mean_ask_volume_historical_list[equity][hour].pop(0)
                    self.equities_and_mid_price_volatility[equity][hour].pop(0)
                    self.equities_and_ticks_in_each_period_historical_list[equity][hour].pop(0)