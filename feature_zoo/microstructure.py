import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..utils import *

class BekkerParkinsonVol:
    def __init__(self,
                 high_atom_name,
                 low_atom_name,
                 clustering_atom_name,
                 window,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.high_atom_name = high_atom_name
        self.low_atom_name = low_atom_name
        self.clustering_atom_name = clustering_atom_name
        self.window = window
        self.rets_form_list = rets_form_list
        self.reference_window = reference_window
        self.reference = reference
        self.standard_scaler = StandardScaler()
        self.windows_buffer_mult = windows_buffer_mult

    def transform(self,
                  atoms,
                  start_feature_data_date,
                  prediction_date,
                  realization_date,
                  reference_id,
                  active_cross_section_series,
                  universe_name):

        params_list = [prediction_date, 
                       start_feature_data_date,
                       self.high_atom_name,
                       self.low_atom_name,
                       self.window,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='BekkerParkinsonVol')

        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.high_atom_name,
                              self.low_atom_name,]
            windows_list = [self.window,
                            self.reference_window]
                            
            (high_atom,
            low_atom,) = get_processed_atoms(
                atoms=atoms,
                atom_name_list=atom_name_list,
                start_feature_data_date=start_feature_data_date,
                prediction_date=prediction_date,
                windows_list=windows_list,
                rets_form_list=self.rets_form_list,
                windows_buffer_mult=self.windows_buffer_mult,
                active_cross_section=active_cross_section_series[
                    prediction_date
                ]
            )

            indicator = high_atom.apply(
                lambda x: self.get_bekker_parkinson_vol(
                    high=x,
                    low=low_atom[x.name],
                    window=self.window,
                )
            ).iloc[-self.reference_window:].replace({np.inf: np.nan, -np.inf: np.nan}).dropna(how='all', axis=1)
            pickle.dump(indicator, open(path_to_file, 'wb'))

        if self.reference == 'cross':
            (sec_to_cluster, 
            cluster_to_sec) = get_clusters(
                prediction_date=prediction_date, 
                clustering_atom=atoms[self.clustering_atom_name],
                active_cross_section=active_cross_section_series[
                    prediction_date
                ],
                universe_name=universe_name,
            )
        else:
            sec_to_cluster = None
            cluster_to_sec = None

        scaled_indicator = get_scaled_indicator(
            indicator=indicator, 
            scaler=self.standard_scaler, 
            reference=self.reference, 
            reference_window=self.reference_window, 
            reference_id=reference_id,
            sec_to_cluster=sec_to_cluster,
            cluster_to_sec=cluster_to_sec,
        )

        return scaled_indicator

    def get_bekker_parkinson_vol(self, high, low, window):
        """
        Get Bekker-Parkinson volatility from gamma and beta in 
        Corwin-Schultz algorithm, (p.286, Snippet 19.2).
        """

        beta = self._get_beta(high, low, window)
        gamma = self._get_gamma(high, low)

        k2 = (8 / np.pi) ** 0.5
        den = 3 - 2 * 2 ** .5
        sigma = (2 ** -0.5 - 1) * beta ** 0.5 / (k2 * den)
        sigma += (gamma / (k2 ** 2 * den)) ** 0.5
        sigma[sigma < 0] = 0

        return sigma

    def _get_beta(self, high, low, window):
        """
        Get beta estimate from Corwin-Schultz algorithm 
        (p.285, Snippet 19.1).
        """

        ret = np.log(high / low)
        high_low_ret = ret ** 2
        beta = high_low_ret.rolling(window=2).sum()
        beta = beta.rolling(window=window).mean()

        return beta

    def _get_gamma(self, high, low):
        """
        Get gamma estimate from Corwin-Schultz algorithm 
        (p.285, Snippet 19.1).
        """

        high_max = high.rolling(window=2).max()
        low_min = low.rolling(window=2).min()
        gamma = np.log(high_max / low_min) ** 2

        return gamma

class BarKyleLambda:
    def __init__(self,
                 close_atom_name,
                 volume_atom_name,
                 clustering_atom_name,
                 window,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.close_atom_name = close_atom_name
        self.volume_atom_name = volume_atom_name
        self.clustering_atom_name = clustering_atom_name
        self.window = window
        self.rets_form_list = rets_form_list
        self.reference_window = reference_window
        self.reference = reference
        self.standard_scaler = StandardScaler()
        self.windows_buffer_mult = windows_buffer_mult

    def transform(self,
                  atoms,
                  start_feature_data_date,
                  prediction_date,
                  realization_date,
                  reference_id,
                  active_cross_section_series,
                  universe_name):

        params_list = [prediction_date, 
                       start_feature_data_date,
                       self.close_atom_name,
                       self.volume_atom_name,
                       self.window,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='BarKyleLambda')

        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.close_atom_name,
                              self.volume_atom_name,]
            windows_list = [self.window,
                            self.reference_window]
                            
            (close_atom,
            volume_atom,) = get_processed_atoms(
                atoms=atoms,
                atom_name_list=atom_name_list,
                start_feature_data_date=start_feature_data_date,
                prediction_date=prediction_date,
                windows_list=windows_list,
                rets_form_list=self.rets_form_list,
                windows_buffer_mult=self.windows_buffer_mult,
                active_cross_section=active_cross_section_series[
                    prediction_date
                ]
            )

            indicator = close_atom.apply(
                lambda x: self.get_bar_based_kyle_lambda(
                    close=x,
                    volume=volume_atom[x.name],
                    window=self.window,
                )
            ).iloc[-self.reference_window:].replace({np.inf: np.nan, -np.inf: np.nan}).dropna(how='all', axis=1)
            pickle.dump(indicator, open(path_to_file, 'wb'))

        if self.reference == 'cross':
            (sec_to_cluster, 
            cluster_to_sec) = get_clusters(
                prediction_date=prediction_date, 
                clustering_atom=atoms[self.clustering_atom_name],
                active_cross_section=active_cross_section_series[
                    prediction_date
                ],
                universe_name=universe_name,
            )
        else:
            sec_to_cluster = None
            cluster_to_sec = None

        scaled_indicator = get_scaled_indicator(
            indicator=indicator, 
            scaler=self.standard_scaler, 
            reference=self.reference, 
            reference_window=self.reference_window, 
            reference_id=reference_id,
            sec_to_cluster=sec_to_cluster,
            cluster_to_sec=cluster_to_sec,
        )

        return scaled_indicator

    def get_bar_based_kyle_lambda(self, close, volume, window):
        """
        Get Kyle lambda from bars data, p.286-288.
        """

        close_diff = close.diff()
        close_diff_sign = np.sign(close_diff)
        # Replace 0 values with previous
        close_diff_sign.replace(0, method='pad', inplace=True) 
        volume_mult_trade_signs = volume * close_diff_sign  # bt * Vt

        return (
            (close_diff 
            / volume_mult_trade_signs).rolling(window=window).mean()
        )

class BarAmihudLambda:
    def __init__(self,
                 close_atom_name,
                 dollar_volume_atom_name,
                 clustering_atom_name,
                 window,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.close_atom_name = close_atom_name
        self.dollar_volume_atom_name = dollar_volume_atom_name
        self.clustering_atom_name = clustering_atom_name
        self.window = window
        self.rets_form_list = rets_form_list
        self.reference_window = reference_window
        self.reference = reference
        self.standard_scaler = StandardScaler()
        self.windows_buffer_mult = windows_buffer_mult

    def transform(self,
                  atoms,
                  start_feature_data_date,
                  prediction_date,
                  realization_date,
                  reference_id,
                  active_cross_section_series,
                  universe_name):

        params_list = [prediction_date, 
                       start_feature_data_date,
                       self.close_atom_name,
                       self.dollar_volume_atom_name,
                       self.window,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='BarAmihudLambda')

        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.close_atom_name,
                              self.dollar_volume_atom_name,]
            windows_list = [self.window,
                            self.reference_window]
                            
            (close_atom,
            dollar_volume_atom,) = get_processed_atoms(
                atoms=atoms,
                atom_name_list=atom_name_list,
                start_feature_data_date=start_feature_data_date,
                prediction_date=prediction_date,
                windows_list=windows_list,
                rets_form_list=self.rets_form_list,
                windows_buffer_mult=self.windows_buffer_mult,
                active_cross_section=active_cross_section_series[
                    prediction_date
                ]
            )

            indicator = close_atom.apply(
                lambda x: self.get_bar_based_amihud_lambda(
                    close=x,
                    dollar_volume=dollar_volume_atom[x.name],
                    window=self.window,
                )
            ).iloc[-self.reference_window:].replace({np.inf: np.nan, -np.inf: np.nan}).dropna(how='all', axis=1)
            pickle.dump(indicator, open(path_to_file, 'wb'))

        if self.reference == 'cross':
            (sec_to_cluster, 
            cluster_to_sec) = get_clusters(
                prediction_date=prediction_date, 
                clustering_atom=atoms[self.clustering_atom_name],
                active_cross_section=active_cross_section_series[
                    prediction_date
                ],
                universe_name=universe_name,
            )
        else:
            sec_to_cluster = None
            cluster_to_sec = None

        scaled_indicator = get_scaled_indicator(
            indicator=indicator, 
            scaler=self.standard_scaler, 
            reference=self.reference, 
            reference_window=self.reference_window, 
            reference_id=reference_id,
            sec_to_cluster=sec_to_cluster,
            cluster_to_sec=cluster_to_sec,
        )

        return scaled_indicator

    def get_bar_based_amihud_lambda(self, close, dollar_volume, window):
        """
        Get Amihud lambda from bars data, p.288-289.
        """
        
        returns_abs = np.log(close / close.shift(1)).abs()

        return (
            (returns_abs 
            / dollar_volume).rolling(window=window).mean()
        )

class BarHasbrouckLambda:
    def __init__(self,
                 close_atom_name,
                 dollar_volume_atom_name,
                 clustering_atom_name,
                 window,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.close_atom_name = close_atom_name
        self.dollar_volume_atom_name = dollar_volume_atom_name
        self.clustering_atom_name = clustering_atom_name
        self.window = window
        self.rets_form_list = rets_form_list
        self.reference_window = reference_window
        self.reference = reference
        self.standard_scaler = StandardScaler()
        self.windows_buffer_mult = windows_buffer_mult

    def transform(self,
                  atoms,
                  start_feature_data_date,
                  prediction_date,
                  realization_date,
                  reference_id,
                  active_cross_section_series,
                  universe_name):

        params_list = [prediction_date, 
                       start_feature_data_date,
                       self.close_atom_name,
                       self.dollar_volume_atom_name,
                       self.window,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='BarHasbrouckLambda')

        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.close_atom_name,
                              self.dollar_volume_atom_name,]
            windows_list = [self.window,
                            self.reference_window]
                            
            (close_atom,
            dollar_volume_atom,) = get_processed_atoms(
                atoms=atoms,
                atom_name_list=atom_name_list,
                start_feature_data_date=start_feature_data_date,
                prediction_date=prediction_date,
                windows_list=windows_list,
                rets_form_list=self.rets_form_list,
                windows_buffer_mult=self.windows_buffer_mult,
                active_cross_section=active_cross_section_series[
                    prediction_date
                ]
            )

            indicator = close_atom.apply(
                lambda x: self.get_bar_based_hasbrouck_lambda(
                    close=x,
                    dollar_volume=dollar_volume_atom[x.name],
                    window=self.window,
                )
            ).iloc[-self.reference_window:].replace({np.inf: np.nan, -np.inf: np.nan}).dropna(how='all', axis=1)
            pickle.dump(indicator, open(path_to_file, 'wb'))

        if self.reference == 'cross':
            (sec_to_cluster, 
            cluster_to_sec) = get_clusters(
                prediction_date=prediction_date, 
                clustering_atom=atoms[self.clustering_atom_name],
                active_cross_section=active_cross_section_series[
                    prediction_date
                ],
                universe_name=universe_name,
            )
        else:
            sec_to_cluster = None
            cluster_to_sec = None

        scaled_indicator = get_scaled_indicator(
            indicator=indicator, 
            scaler=self.standard_scaler, 
            reference=self.reference, 
            reference_window=self.reference_window, 
            reference_id=reference_id,
            sec_to_cluster=sec_to_cluster,
            cluster_to_sec=cluster_to_sec,
        )

        return scaled_indicator

    def get_bar_based_hasbrouck_lambda(self, close, dollar_volume, window):
        """
        Get Hasbrouck lambda from bars data, p.289-290.
        """

        log_ret = np.log(close / close.shift(1))
        log_ret_sign = np.sign(log_ret).replace(0, method='pad')

        signed_dollar_volume_sqrt = log_ret_sign * np.sqrt(dollar_volume)

        return (
            (log_ret 
            / signed_dollar_volume_sqrt).rolling(window=window).mean()
        )

class BarVPIN:
    def __init__(self,
                 volume_atom_name,
                 buy_volume_atom_name,
                 clustering_atom_name,
                 window,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.volume_atom_name = volume_atom_name
        self.buy_volume_atom_name = buy_volume_atom_name
        self.clustering_atom_name = clustering_atom_name
        self.window = window
        self.rets_form_list = rets_form_list
        self.reference_window = reference_window
        self.reference = reference
        self.standard_scaler = StandardScaler()
        self.windows_buffer_mult = windows_buffer_mult

    def transform(self,
                  atoms,
                  start_feature_data_date,
                  prediction_date,
                  realization_date,
                  reference_id,
                  active_cross_section_series,
                  universe_name):

        params_list = [prediction_date, 
                       start_feature_data_date,
                       self.volume_atom_name,
                       self.buy_volume_atom_name,
                       self.window,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='BarVPIN')

        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.volume_atom_name,
                              self.buy_volume_atom_name,]
            windows_list = [self.window,
                            self.reference_window]
                            
            (volume_atom,
            buy_volume_atom,) = get_processed_atoms(
                atoms=atoms,
                atom_name_list=atom_name_list,
                start_feature_data_date=start_feature_data_date,
                prediction_date=prediction_date,
                windows_list=windows_list,
                rets_form_list=self.rets_form_list,
                windows_buffer_mult=self.windows_buffer_mult,
                active_cross_section=active_cross_section_series[
                    prediction_date
                ]
            )

            indicator = volume_atom.apply(
                lambda x: self.get_vpin(
                    volume=x,
                    buy_volume=buy_volume_atom[x.name],
                    window=self.window,
                )
            ).iloc[-self.reference_window:].replace({np.inf: np.nan, -np.inf: np.nan}).replace(
                {np.inf: np.nan, -np.inf: np.nan}
            ).dropna(how='all', axis=1)
            pickle.dump(indicator, open(path_to_file, 'wb'))

        if self.reference == 'cross':
            (sec_to_cluster, 
            cluster_to_sec) = get_clusters(
                prediction_date=prediction_date, 
                clustering_atom=atoms[self.clustering_atom_name],
                active_cross_section=active_cross_section_series[
                    prediction_date
                ],
                universe_name=universe_name,
            )
        else:
            sec_to_cluster = None
            cluster_to_sec = None

        scaled_indicator = get_scaled_indicator(
            indicator=indicator, 
            scaler=self.standard_scaler, 
            reference=self.reference, 
            reference_window=self.reference_window, 
            reference_id=reference_id,
            sec_to_cluster=sec_to_cluster,
            cluster_to_sec=cluster_to_sec,
        )

        return scaled_indicator

    def get_vpin(self, volume, buy_volume, window):
        """
        Get Volume-Synchronized Probability of Informed Trading (VPIN) 
        from bars, p. 292-293.
        """

        sell_volume = volume - buy_volume
        volume_imbalance = abs(buy_volume - sell_volume)

        return volume_imbalance.rolling(window=window).mean() / volume
