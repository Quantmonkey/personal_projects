from re import T
import numpy as np
from scipy.linalg import fractional_matrix_power
from sklearn.linear_model import LinearRegression

def box_tiao_canonical_decomposition(prices_array,
                                     max_lag=1):
    
    B_sqrt_inv = __get_B_sqrt_inv(prices_array=prices_array)
    A = __get_A(prices_array=prices_array, max_lag=max_lag)
    D = np.matmul(np.matmul(B_sqrt_inv, A), B_sqrt_inv)

    eigenvalues, eigenvectors = np.linalg.eigh(D)

    lambdas = eigenvalues
    hedge_ratios = np.matmul(B_sqrt_inv, eigenvectors)

    return lambdas, hedge_ratios

def __get_B_sqrt_inv(prices_array):

    B = __get_expected_dyadic_prod(prices_array)
    B_sqrt = fractional_matrix_power(B, 0.5)
    B_sqrt_inverse = np.linalg.inv(B_sqrt)

    return B_sqrt_inverse

def __get_A(prices_array, max_lag):
    
    X_prices_array_lagged = np.concatenate(
        [prices_array[max_lag-lag: -lag, :]
        for lag in range(1, max_lag+1)],
        axis=1
    )

    price_series_predictions = []
    for price_series_index in range(prices_array.shape[1]):
        y_price_array = prices_array[max_lag:, price_series_index]
        ls_model = LinearRegression().fit(X_prices_array_lagged,
                                          y_price_array)
        price_series_prediction = ls_model.predict(X_prices_array_lagged)
        price_series_predictions.append(price_series_prediction)

    q_predictions_array = np.asarray(price_series_predictions).T

    return __get_expected_dyadic_prod(q_predictions_array)

def __get_expected_dyadic_prod(V):

    expected_dyadic_product = (1./V.shape[0])*np.matmul(V.T, V)

    return expected_dyadic_product