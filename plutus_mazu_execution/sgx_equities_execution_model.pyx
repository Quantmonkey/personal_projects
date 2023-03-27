import pandas as pd

from helper_models.general.exception_handler import ExceptionHandler
from helper_models.general.sampler import Sampler
from helper_models.slice.child_order_manager import ChildOrderManager
from helper_models.slice.sgx_equities_vwap import SGXEquitiesVwap
from helper_models.data.historical_downloader_new import HistoricalDownloader
from tqdm import tqdm
exception_handler = ExceptionHandler()
INTEGER_MAX = 2147483600


class SGXEquitiesExecutionModel:
    """
    Execution model is responsible for executing orders emitted from Portfolio model.

    The goal is to use non-naive execution strategies to minimise signalling, and to do that
    a vwap model is used to slice our order into child orders and a child order manager
    is responsible for executing the child orders.
    """
    def __init__(self, strategy_config):
        self.strategy_config = strategy_config
        self.verbose = strategy_config.verbose
        self.sampler = Sampler(strategy_config)

        # internal data structures
        self.orders_to_fulfill = {}  # full order vector, received from portfolio model.
        self.child_orders = {}  # orders_to_fulfill sliced into 10 child orders.
        self.partial_order_vector = {}  # partial orders to fulfill, updated every 30s by adding a child order.
        self.equities_and_current_limit_orders = {}
        self.pending_orders_to_be_submitted = []
        self.equities_and_order_status = {}
        self.unique_int_id = 0  # Used to provide each equity with a unique tag.

        # data structures to keep track of time
        self.previous_order_number = None
        self.previous_update_datetime = None
        self.time_order_received = None

        # helper models
        self.historical_model = HistoricalDownloader(strategy_config)
        self.vwap_model = SGXEquitiesVwap(strategy_config)
        self.child_order_manager = ChildOrderManager(strategy_config)

        # historical data structures
        self.equities_and_daily_tick_data = {}
        self.equities_and_daily_trade_data = {}

        # data structures propagated from data model
        self.current_time = None
        self.latest_tick = None
        self.latest_trade = None

        # data structures propagated from universe model
        self.equities_and_id_for_data_collection = {}
        self.equities_and_id_for_trading = {}
        self.equities_and_tick_size = {}
        self.pairs_of_equities = {}

        # lock to prevent sending more orders until an order status is received
        self.waiting_for_order_status = False

    def initialise_data_and_models(self, time_now):
        """
        Initialises all data structures and models required for execution model to function.
        This is called each time universe reset.

        Args:
            time_now (datetime.datetime): used to download historical data before time_now
        """
        self.initialise_data_structures()
        self.initialise_vwap_model(time_now)
        self.initialise_child_order_manager(time_now)

    def initialise_data_structures(self):
        """ Initialises order vector, daily tick and daily trade data.
        """
        self.partial_order_vector = {equity: 0 for equity in self.equities_and_id_for_trading}
        self.equities_and_daily_tick_data = {equity: [] for equity in self.equities_and_id_for_trading}
        self.equities_and_daily_trade_data = {equity: [] for equity in self.equities_and_id_for_trading}

    @exception_handler.except_initialise_vwap_model
    def initialise_vwap_model(self, time_now):
        """
        Initialises Vwap by downloading historical trade data. Vwap will preprocess volume from trade data to calculate 5 minute
        and 30 second buckets, in order to determine how to slice orders. The downloading and updating is done dynamically (one day at a time)
        to reduce RAM usage.

        Args:
            time_now (datetime.datetime): used to download historical data before time_now
        """
        # initialise vwap model with universe
        self.vwap_model.initialise_data_structures(self.equities_and_id_for_trading.keys())

        self.verbose.orate_download_trade_data_for_vwap(time_now)
        for day in tqdm(range(self.strategy_config.DOWNLOAD_HISTORICAL_DATA_FOR_EXECUTION_PERIOD), colour='blue'):
            now = time_now - pd.Timedelta(days=self.strategy_config.DOWNLOAD_HISTORICAL_DATA_FOR_EXECUTION_PERIOD - day - 1)
            equities_and_short_interval_trade_dfs = \
                self.historical_model.download_historical_trade_data(self.equities_and_id_for_trading,
                                                                     now, 1, resample_interval='30s',
                                                                     columns=['last_size'], mute=True)
            equities_and_long_interval_trade_dfs = \
                self.historical_model.download_historical_trade_data(self.equities_and_id_for_trading,
                                                                     now, 1, resample_interval='5m',
                                                                     columns=['last_size'], mute=True)

            self.vwap_model.collect_volume_data(equities_and_short_interval_trade_dfs, equities_and_long_interval_trade_dfs)

        self.vwap_model.process_volume_data()
        self.vwap_model.generate_weight()

    @exception_handler.except_initialise_child_order_manager
    def initialise_child_order_manager(self, time_now):
        """
        Initialises Child order manager by downloading historical tick data. COM will preprocess the tick data to calculate metrics
        such as volatility, price spread widths, frequency of ticks etc. which will be used to determine child order execution logic.
        The downloading and updating is done dynamically (one day at a time) to reduce RAM usage.

        Args:
            time_now (datetime.datetime): used to download historical data before time_now
        """
        # DATA STRUCTURES SHARED BY REFERENCE (allows both execution model and child order manager to modify these data structures)
        self.child_order_manager.receive_data_structures_from_execution_model({"order vector": self.partial_order_vector,
                                                                               "equities and order status": self.equities_and_order_status,
                                                                               "equities and limit orders": self.equities_and_current_limit_orders,
                                                                               "trading universe": self.equities_and_id_for_trading,
                                                                               "equities and tick size": self.equities_and_tick_size})
        self.child_order_manager.initialise_data_structures()

        # Download historical tick data and pass to child order manager model
        self.verbose.orate_download_tick_data_for_child_order_manager(time_now)
        for day in tqdm(range(self.strategy_config.DOWNLOAD_HISTORICAL_DATA_FOR_EXECUTION_PERIOD), colour='cyan'):
            now = time_now - pd.Timedelta(days=self.strategy_config.DOWNLOAD_HISTORICAL_DATA_FOR_EXECUTION_PERIOD - day - 1)

            counts_per_completion_period_dfs = \
                self.historical_model.download_historical_tick_data(self.equities_and_id_for_trading, now, 1,
                                                                    resample_interval=self.strategy_config.CHILD_ORDER_COMPLETION_PERIOD,
                                                                    columns=['count'], mute=True)

            volumes_and_counts_dfs = \
                self.historical_model.download_historical_tick_data(self.equities_and_id_for_trading, now, 1,
                                                                    resample_interval='1h',
                                                                    columns=['vol_bid', 'vol_ask', 'count'],
                                                                    mute=True)

            mid_prices_and_price_spreads_dfs = \
                self.historical_model.download_historical_tick_aggregation(self.equities_and_id_for_trading,
                                                                           now, 1, '1h', mute=True)

            self.child_order_manager.collect_and_process_tick_data(counts_per_completion_period_dfs,
                                                                   volumes_and_counts_dfs,
                                                                   mid_prices_and_price_spreads_dfs)

        for equity in self.equities_and_id_for_trading.keys():
            self.child_order_manager.get_all_benchmarks(equity)

        self.child_order_manager.get_wide_spread_threshold()

    def recalculate_new_metrics_for_vwap_and_com(self, time_now):
        """
        TODO
        """
        self.verbose.orate_recalculate_new_metrics_for_vwap_and_com(time_now)

        # Downloading new data for, and updating vwap model
        equities_and_short_interval_trade_dfs = \
            self.historical_model.download_historical_trade_data(self.equities_and_id_for_trading,
                                                                 time_now, 1, resample_interval='30s',
                                                                 columns=['last_size'], mute=True)

        equities_and_long_interval_trade_dfs = \
            self.historical_model.download_historical_trade_data(self.equities_and_id_for_trading,
                                                                 time_now, 1, resample_interval='5m',
                                                                 columns=['last_size'], mute=True)

        self.vwap_model.collect_volume_data(equities_and_short_interval_trade_dfs, equities_and_long_interval_trade_dfs, updating=True)
        self.vwap_model.process_volume_data()
        self.vwap_model.generate_weight()

        # Downloading new data for, and updating child order manager
        counts_per_completion_period_dfs = \
            self.historical_model.download_historical_tick_data(self.equities_and_id_for_trading, time_now, 1,
                                                                resample_interval=self.strategy_config.CHILD_ORDER_COMPLETION_PERIOD,
                                                                columns=['count'], mute=True)

        volumes_and_counts_dfs = \
            self.historical_model.download_historical_tick_data(self.equities_and_id_for_trading, time_now, 1,
                                                                resample_interval='1h',
                                                                columns=['vol_bid', 'vol_ask', 'count'],
                                                                mute=True)

        mid_prices_and_price_spreads_dfs = \
            self.historical_model.download_historical_tick_aggregation(self.equities_and_id_for_trading,
                                                                       time_now, 1, '1h', mute=True)

        self.child_order_manager.collect_and_process_tick_data(counts_per_completion_period_dfs,
                                                               volumes_and_counts_dfs,
                                                               mid_prices_and_price_spreads_dfs,
                                                               updating=True)

        for equity in self.equities_and_id_for_trading.keys():
            self.child_order_manager.get_all_benchmarks(equity)

        self.child_order_manager.get_wide_spread_threshold()

    @exception_handler.except_slice_order
    def slice_order(self, order_vector):
        # slice order vector into 10 child orders
        if order_vector is not None:
            if any([order_vector[equity] != 0 for equity in order_vector]):
                self.orders_to_fulfill = order_vector
                child_orders = self.vwap_model.slice_order(order_vector, self.current_time)

                # Filter out non zero orders to improve visibility when printing.
                filtered_child_orders = {}
                for order_number in range(10):
                    filtered_child_orders[order_number] = {equity: position for equity, position in child_orders[order_number].items() if position != 0}
                self.verbose.orate_child_orders(self.current_time, filtered_child_orders)

                return child_orders
        else:
            return None

    def execute_new_order_vectors(self, child_orders):
        """
        Main logic for executing orders. Every 30 seconds, a new child order will be executed, by adding the child order
        to the partial order vector. Within this 30 seconds interval, child order manager will determine how to further
        split the child order based on its calculated metrics on market conditions.

        Args:
            child_orders Dict({order_number (int): order (Dict)}): NESTED DATA STRUCTURE

                order Dict({equity_symbol (str): position (int)})

            ### if no order vector is emitted from Portfolio Model, child_orders will be None instead.

        TODO: Checks and ensures that we are at least keeping up to a TWAP schedule, sending MO to catch up if needed.
        """
        # This happens when Portfolio model emits an order vector. Time order received is immediately updated
        # and the next 5 min schedule for executing these child orders will be planned out.
        if child_orders is not None:
            self.time_order_received = self.current_time
            self.child_orders = child_orders
            self.previous_order_number = None

        if self.time_order_received is not None:
            order_number = (self.current_time - self.time_order_received).total_seconds() // 30

            # Ensure that we only update child order vector once every 30 seconds
            if order_number <= 9 and order_number != self.previous_order_number:
                self.previous_order_number = order_number
                # Add the child order to our partial order vector
                for equity, quantity in self.child_orders[order_number].items():
                    self.partial_order_vector[equity] += quantity

                del self.child_orders[order_number]

            # If this is the last order, go into chasing mode, sending market orders to clear remaining orders.
            if order_number > 9:
                self.time_order_received = None
                self.chasing()
            else:
                # Pass responsibility of deciding when to execute the next order to Child Order Manager.
                order = self.child_order_manager.record_ticks_and_check_if_need_to_send_order(self.latest_tick, self.current_time)
                self.create_pending_order(order)

    def reschedule_order_vectors(self, child_orders):
        """
        This is called after loading a pickled strategy. We will create a new schedule to execute
        any remaining orders

        Args:
            child_orders Dict({order_number (int): order (Dict)}): NESTED DATA STRUCTURE

                order Dict({equity_symbol (str): position (int)})

            ### if no order vector is input from the user, child_orders will be None instead.
        """
        if child_orders is not None:
            self.time_order_received = self.current_time
            self.child_orders = child_orders
            self.previous_order_number = None

    def create_pending_order(self, order):
        """
        Creates a pending order (Algotrader order object) from a dictionary emitted from
        Child order manager. Pending orders will be appended to a list (queue) and will be sent
        one at a time.

        Args:
            order Dict({order_metrics: values}): various information for the order (side, quantity, security_id etc.)
        """
        # create a pending order only when quantity is non zero.
        if order["qty"] != 0:
            if self.unique_int_id < INTEGER_MAX:
                self.unique_int_id += 1
            else:
                self.unique_int_id = 0

            equity = order["equity"]
            pending_order = self.strategy_config.ALGOTRADER_LIMIT_ORDER(side=order['side'],
                                                                        quantity=order["qty"],
                                                                        security_id=self.equities_and_id_for_trading[equity],
                                                                        account_id=self.strategy_config.TT_ACCOUNT_ID,
                                                                        strategy_id=self.strategy_config.STRATEGY_ID,
                                                                        limit=order["limit_price"],
                                                                        tif=order["tif"],
                                                                        int_id=f"{self.strategy_config.STRATEGY_NAME}_{equity}_{self.unique_int_id}_{order['side']}")

            self.pending_orders_to_be_submitted.append((equity, pending_order))

    def chasing(self):
        """
        This function will be called when we
        Called when we reached the last child order. Send market orders to clear remaining quantity required.
        """
        # First, we need to cancel any remaining limit orders that we have
        for equity, order_status in self.equities_and_order_status.copy().items():
            if order_status.remaining_quantity != 0:
                side = 1 if 'BUY' in order_status.int_id else -1
                self.partial_order_vector[equity] += side * int(order_status.remaining_quantity)
                self.strategy_config.ALGOTRADER_CANCEL_ORDER_FUNCTION(order_status.int_id)
                del self.equities_and_order_status[equity]

        for equity, qty in self.partial_order_vector.items():
            if qty != 0:
                if self.unique_int_id < INTEGER_MAX:
                    self.unique_int_id += 1
                else:
                    self.unique_int_id = 0

                side = "BUY" if qty > 0 else "SELL"
                security_id = self.equities_and_id_for_trading[equity]

                pending_order = self.strategy_config.ALGOTRADER_MARKET_ORDER(side=side,
                                                                             quantity=abs(qty),
                                                                             security_id=security_id,
                                                                             account_id=self.strategy_config.TT_ACCOUNT_ID,
                                                                             strategy_id=self.strategy_config.STRATEGY_ID,
                                                                             int_id=f"{self.strategy_config.STRATEGY_NAME}_{equity}_{self.unique_int_id}_{side}")

                self.pending_orders_to_be_submitted.append((equity, pending_order))

    def send_pending_orders(self):
        """
        TODO
        """
        if self.pending_orders_to_be_submitted and not self.waiting_for_order_status:
            # Pops the next equity and pending order out of the queue
            equity, pending_order = self.pending_orders_to_be_submitted.pop(0)

            # Lock to prevent any more orders from being sent until an order status is received.
            self.lock_execution_model()

            # Checks if the last order for this equity is still incomplete,
            # and cancel the order if it is. This is to ensure that we won't have
            # stagnant limit orders that are placed on the market and never canceled.
            if equity in self.equities_and_order_status and self.equities_and_order_status[equity].status != 'CANCELED':
                # Canceling orders may fail in the (very rare) event that the order is
                # filled when we are still halfway in the main event loop
                try:
                    self.strategy_config.ALGOTRADER_CANCEL_ORDER_FUNCTION(self.equities_and_order_status[equity].int_id)
                    side = 1 if 'BUY' in self.equities_and_order_status[equity].int_id else -1
                    self.partial_order_vector[equity] += side * int(self.equities_and_order_status[equity].remaining_quantity)
                    del self.equities_and_order_status[equity]
                except:
                    pass

            # Sends a new order for this security
            self.equities_and_current_limit_orders[equity] = pending_order
            self.strategy_config.ALGOTRADER_SEND_ORDER_FUNCTION(pending_order)

            # Update order vector
            side = 1 if pending_order.side == 'BUY' else -1
            self.partial_order_vector[equity] -= int(pending_order.quantity * side)

    def receive_order_status_and_update(self, order_status):
        """Helper function to receive order status.
        """
        for equity in list(self.equities_and_current_limit_orders.keys()):
            # assigns order status object to the current equities
            if order_status.int_id == self.equities_and_current_limit_orders[equity].int_id:
                self.equities_and_order_status[equity] = order_status
                # Order status has been received, so the lock can be released
                if order_status.status == 'OPEN':
                    self.unlock_execution_model()
                break

    def lock_execution_model(self):
        """TODO
        """
        self.waiting_for_order_status = True
        self.child_order_manager.waiting_for_order_status = True

    def unlock_execution_model(self):
        """TODO
        """
        self.waiting_for_order_status = False
        self.child_order_manager.waiting_for_order_status = False

    def clear_all_order_data_structures(self):
        """
        TODO
        """
        self.verbose.orate_cancel_all_orders(self.current_time, list(self.equities_and_current_limit_orders.keys()))
        for equity, order in list(self.equities_and_current_limit_orders.items()):
            # Canceling orders may fail if the order has already been executed when the strategy was taken down
            try:
                self.strategy_config.ALGOTRADER_CANCEL_ORDER_FUNCTION(order.int_id)
                del self.equities_and_current_limit_orders[equity]
                del self.equities_and_order_status[equity]
            except:
                pass

    def receive_subscribed_universe(self, list_of_data):
        """Receive subscribed equity and currencies from universe model
        """
        self.equities_and_id_for_data_collection = list_of_data[0]
        self.equities_and_id_for_trading = list_of_data[1]
        self.equities_and_tick_size = list_of_data[2]
        self.pairs_of_equities = list_of_data[3]

    def receive_data_from_data_model(self, data):
        """ Receive latest data from data model, every tick and bar
        """
        self.current_time = data['current time']
        self.child_order_manager.equities_and_latest_tick = data['equities and latest tick']
        self.latest_tick = data['latest tick']
        self.latest_trade = data['latest trade']
