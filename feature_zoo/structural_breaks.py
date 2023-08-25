import pickle
import numpy as np
import pandas as pd
from ta.trend import *
from sklearn.preprocessing import StandardScaler

from numba import jit, njit, prange

from ..utils import *

class SupremumDickeyFullerChowStatTest:
    def __init__(self,
                 close_atom_name,
                 clustering_atom_name,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.close_atom_name = close_atom_name
        self.clustering_atom_name = clustering_atom_name
        self.min_num_samples = reference_window
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
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='SupremumDickeyFullerChowStatTest')

        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.close_atom_name,]
            windows_list = [self.reference_window]
                            
            (close_atom,) = get_processed_atoms(
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
                lambda x: self.get_series_sdfc(
                    close=x,
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
    
    def get_series_sdfc(self,
                        close):
        indices = np.arange(
            self.min_num_samples, close.shape[0] - self.min_num_samples
        )
        series_diff = close.ffill().diff().dropna()
        series_lag = close.ffill().shift(1).dropna()
        index = close.index.values[
            self.min_num_samples: 
            close.shape[0] - self.min_num_samples
        ]
        try:
            dfc_array = get_dfc_array(
                indices,
                np.ascontiguousarray(
                    series_diff.values.reshape(-1, 1), 
                    dtype=np.float64
                ),
                np.ascontiguousarray(
                    series_lag.values.reshape(-1, 1), 
                    dtype=np.float64
                ),
            )
        except:
            dfc_array = np.empty(index.shape)
            dfc_array[:] = np.nan

        return pd.Series(
            dfc_array,
            index=index
        )

@jit(parallel=True, nopython=True)
def get_dfc_array(indices, y_diff, y_lag):
    """Returns the Chow-Type Dickey-Fuller t-values.
    """
    
    dfc_array = np.empty_like(indices, dtype=np.float64)
    for i in prange(len(indices)):
        dummy_var = np.ones_like(y_lag)
        # D_t* indicator: before t* D_t* = 0
        dummy_var[: indices[i]] = 0  
        X = y_lag * dummy_var
        beta_hat, beta_var = get_beta_and_beta_var(
            X,
            y_diff,
        )
        dfc_array[i] = beta_hat[0, 0] / np.sqrt(beta_var[0, 0])

    return dfc_array


@njit
def get_beta_and_beta_var(X, y):
    """
    Returns the OLS estimates of the coefficients and the variance of 
    the coefficients.
    """
    Xy = np.dot(X.T, y)
    XX = np.dot(X.T, X)
    try:
        XX_inv = np.linalg.inv(XX)
    except:
        XX_inv = np.linalg.pinv(XX)

    beta_hat = np.dot(np.ascontiguousarray(XX_inv), Xy)
    err = y - np.dot(X, beta_hat)
    beta_hat_var = np.dot(err.T, err) / (X.shape[0] - X.shape[1]) * XX_inv

    return beta_hat, beta_hat_var

class ChuStinchcombeWhiteStatTest:
    def __init__(self,
                 close_atom_name,
                 clustering_atom_name,
                 side_test,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.close_atom_name = close_atom_name
        self.clustering_atom_name = clustering_atom_name
        self.side_test = side_test
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
                       self.side_test,
                       self.close_atom_name,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='ChuStinchcombeWhiteStatTest')

        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.close_atom_name,]
            windows_list = [self.reference_window]
                            
            (close_atom,) = get_processed_atoms(
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
                lambda x: self.get_series_cscw(
                    close=x,
                    side_test=self.side_test,
                )
            ).iloc[-self.reference_window:].replace(
                {np.inf: np.nan, -np.inf: np.nan,}
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

    def get_series_cscw(self, close, side_test):
        """Runs the Chu-Stinchcombe-White stat test and returns
        a dataframe with the test statistics and corresponding 
        critical values.
        """

        t_indices = np.arange(2, len(close) + 1)
        S_n_t_array = get_test_statistics_array(
            side_test, t_indices, np.ascontiguousarray(close.values)
        )

        return pd.DataFrame(
            S_n_t_array,
            index=close.index[1:],
            columns=['statistic', 'critical_value'],
        )['statistic']

@jit(parallel=True, nopython=True)
def get_test_statistics_array(side_test, t_indices, time_series_array):
    """Returns an array with the test statistics of
    the Chu-Stinchcombe-White and the critical values
    """
    b_alpha_5_pct = 4.6  # 4.6 is b_a estimate derived via Monte-Carlo
    test_statistics_array = np.empty(
        (t_indices.shape[0], 2), 
        dtype=np.float64
    )
    # outer loops goes over all t
    for i in prange(len(t_indices)):

        # compute variance
        t = t_indices[i]
        array_t = time_series_array[:t]
        sigma_squared_t = np.sum(np.square(np.diff(array_t))) / (t - 1)

        # init supremum vals
        max_S_n_t_value = -np.inf
        max_S_n_t_critical_value = np.nan  # Corresponds to c_alpha[n,t]

        y_t = array_t[t - 1]
        # inner loop goes over all n between 1 and t
        for j in prange(len(array_t) - 1):
            # compute test statistic
            n = j + 1
            y_n = time_series_array[j]
            y_t_y_n_diff = get_y_t_y_n_diff(side_test, y_t, y_n)
            S_n_t = y_t_y_n_diff / np.sqrt(sigma_squared_t * (t - n))

            # check if new val is better than supremum
            # if so compute new critical value
            if S_n_t > max_S_n_t_value:
                max_S_n_t_value = S_n_t
                max_S_n_t_critical_value = np.sqrt(b_alpha_5_pct + np.log(t - n))
        # store result of iteration
        test_statistics_array[i, :] = np.array(
            [max_S_n_t_value, max_S_n_t_critical_value], dtype=np.float64
        )

    return test_statistics_array

@njit
def get_y_t_y_n_diff(side_test, y_t, y_n):
    """Returns the difference between y_t and y_n given
    a test specification.
    """

    if side_test == 'one_sided_positive':
        values_diff = y_t - y_n
    elif side_test == 'one_sided_negative':
        values_diff = y_n - y_t
    else:
        values_diff = abs(y_t - y_n)

    return values_diff

class SupremumAugmentedDickeyFullerStatTest:
    def __init__(self,
                 close_atom_name,
                 clustering_atom_name,
                 model,
                 lags,
                 phi,
                 rets_form_list,
                 reference,
                 reference_window,
                 windows_buffer_mult):
        self.close_atom_name = close_atom_name
        self.clustering_atom_name = clustering_atom_name
        self.add_constant = True
        self.model =  model
        self.lags = lags
        self.phi = phi
        self.min_num_samples = reference_window
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
                       self.model,
                       self.lags,
                       self.phi,
                       self.close_atom_name,
                       self.rets_form_list,
                       self.reference_window]

        path_to_file = get_path_to_file(
            params_list=params_list, 
            ABS_PATH_TO_CACHE=ABS_PATH_TO_CACHE+universe_name+'/', 
            unique_name='SupremumAugmentedDickeyFullerStatTest')

        indicator = get_pickled_indicator(path_to_file)
        if indicator.empty:
            atom_name_list = [self.close_atom_name,]
            windows_list = [self.reference_window]
                            
            (close_atom,) = get_processed_atoms(
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
                lambda x: self.get_series_sadf(
                    close=x,
                )
            ).iloc[-self.reference_window:].replace(
                {np.inf: np.nan, -np.inf: np.nan,}
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

    def get_series_sadf(self, close):
        """
        Runs the Supremum Augmented Dickey Fuller test and returns a
        series with SADF statistics.
        When the model specification is for sub- and super-martingale
        then a series with SMT statistics is returned. See page 260
        for additional details.
        """
        X, y = _get_X_y(close, self.model, self.lags, self.add_constant)
        indices = y.index[self.min_num_samples:]

        try:
            sadf_array = _sadf_outer_loop(
                X=X.values.astype(np.float64),
                y=y.values.astype(np.float64).reshape(-1, 1),
                min_num_samples=self.min_num_samples,
                model=self.model,
                phi=self.phi,
            )
        except:
            sadf_array = np.empty(len(indices))
            sadf_array[:] = np.nan

        return pd.Series(sadf_array, index=indices)

@jit(parallel=False, nopython=True)
def _get_sadf_at_t(X, y, min_num_samples, model, phi):
    """
    Snippet 17.2, page 258. SADF's Inner Loop (get SADF value at t)
    """

    adf_array = np.empty(y.shape[0] - min_num_samples + 1)

    # inner loop starts from 1 to t - tau see page 253
    for start in prange(0, y.shape[0] - min_num_samples + 1):
        X_ = np.ascontiguousarray(X[start:])
        y_ = np.ascontiguousarray(y[start:])

        b_mean_, b_var_ = get_beta_and_beta_var(X_, y_)

        current_adf = b_mean_[0, 0] / b_var_[0, 0] ** 0.5

        # if the test specification is a sub- or super-martingale test
        # adjust the test statistic as described on page 260.
        if model[:2] == "sm":
            t = y.shape[0]
            t0 = start + 1  # t0 index starts from 1 to t - tau (page 260)
            current_adf = np.abs(current_adf) / ((y.shape[0] - (t - t0)) ** phi)

        adf_array[start] = current_adf

    return np.max(adf_array)


def _get_X_y_standard_specification(series, lags):
    """
    Returns the matrix with features X and y for the
    standard specifcation without constant as described on
    page 252.
    """

    series_diff = series.diff().dropna()
    X = _lag_df(series_diff.to_frame(), lags).dropna()
    X["y_lagged"] = series.shift(1).loc[X.index]  # add y_(t-1) column
    y = series_diff.loc[X.index]

    return X, y

def _get_X_y(series, model, lags, add_const):
    """
    Snippet 17.2, page 258-259. Preparing The Datasets
    """

    if model == "no_trend":
        X, y = _get_X_y_standard_specification(series, lags)
        beta_column = "y_lagged"
        if add_const:
            X["const"] = 1
    elif model == "linear":
        X, y = _get_X_y_standard_specification(series, lags)
        X["trend"] = np.arange(
            1, X.shape[0] + 1
        )  # Add t to the model (0, 1, 2, 3, 4, 5, .... t)
        beta_column = (
            "y_lagged"  # Column which is used to estimate test beta statistics
        )
        if add_const:
            X["const"] = 1
    elif model == "quadratic":
        X, y = _get_X_y_standard_specification(series, lags)
        X["trend"] = np.arange(
            1, X.shape[0] + 1
        )  # Add t to the model (0, 1, 2, 3, 4, 5, .... t)
        X["quad_trend"] = X["trend"] ** 2  # Add t^2 to the model (0, 1, 4, 9, ....)
        beta_column = (
            "y_lagged"  # Column which is used to estimate test beta statistics
        )
        if add_const:
            X["const"] = 1
    elif model == "sm_poly_1":
        y = series.copy()
        X = pd.DataFrame(index=y.index)
        X["const"] = 1
        X["trend"] = np.arange(1, X.shape[0] + 1)
        X["quad_trend"] = X["trend"] ** 2
        beta_column = "quad_trend"
    elif model == "sm_poly_2":
        y = np.log(series.copy())
        X = pd.DataFrame(index=y.index)
        X["const"] = 1
        X["trend"] = np.arange(1, X.shape[0] + 1)
        X["quad_trend"] = X["trend"] ** 2
        beta_column = "quad_trend"
    elif model == "sm_exp":
        y = np.log(series.copy())
        X = pd.DataFrame(index=y.index)
        X["const"] = 1
        X["trend"] = np.arange(1, X.shape[0] + 1)
        beta_column = "trend"
    elif model == "sm_power":
        y = np.log(series.copy())
        X = pd.DataFrame(index=y.index)
        X["const"] = 1
        X["log_trend"] = np.log(np.arange(1, X.shape[0] + 1))
        beta_column = "log_trend"
    else:
        raise ValueError("Unknown model")

    # Move y_lagged column to the front for further extraction
    columns = list(X.columns)
    columns.insert(0, columns.pop(columns.index(beta_column)))
    X = X[columns].ffill().bfill().dropna(how='all', axis=1)
    y = y.ffill().bfill().dropna(how='all')

    assert (
        ~X.isna().any().any() and ~y.isna().any().any()
    ), f"The constructed X and y contain NaNs based on the model specification {model}."

    return X, y


def _lag_df(df, lags):
    """
    Snipet 17.3, page 259. Apply Lags to DataFrame
    """
    df_lagged = pd.DataFrame()
    if isinstance(lags, int):
        lags = range(1, lags + 1)
    else:
        lags = [int(lag) for lag in lags]

    for lag in lags:
        temp_df = df.shift(lag).copy(deep=True)
        temp_df.columns = [
            str(i) 
            + "_diff_lag_" 
            + str(lag) 
            for i in temp_df.columns]
        df_lagged = df_lagged.join(temp_df, how="outer")

    return df_lagged


@jit(parallel=True, nopython=True)
def _sadf_outer_loop(X, y, min_num_samples, model, phi,):
    """Runs the SADF outer loop, i.e for each t in T.
    """

    sadf_array = np.empty(
        (y.shape[0] 
        - min_num_samples), 
    dtype=np.float64)
    for index in prange(min_num_samples, y.shape[0]):
        sadf_array[index - min_num_samples] = _get_sadf_at_t(
            X[: (index + 1), :], y[: (index + 1), :], 
            min_num_samples, model, phi
        )

    return sadf_array



