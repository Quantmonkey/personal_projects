%%cython

import numpy as np
import pandas as pd
from scipy.optimize import minimize

def minimize_DF(prices, weights_init=None):
    """Cointegration weights weightsy direct minimization of the Dickey Fuller statistic.
    
    Args:
        np.ndarray prices : A numpy 2D array of shape (T+1, N).
        np.ndarray weights_init: Initial guess of an optimal hedge ratio
    
    Returns:
        np.ndarray weights: The optimal hedge ratio
    """
    
    try:
        weights_0 = get_weights_0(weights_init)

        (cov_p,
         cov_differenced,
         cov_lagged,
         cov_differenced_lagged,
         cov_lagged_differenced) = get_prices_cov_matrices(prices)

        opt_result = minimize(
            lambda weights: get_DF(weights, prices),
            x0=weights_0,
            method='trust-krylov',
            jac=lambda weights: get_DF_gradient(
                weights, 
                prices, 
                cov_lagged,
                cov_differenced,
                cov_differenced_lagged,
                cov_lagged_differenced
            ),
            hess=lambda weights: get_hessian_DF(
                weights, 
                prices, 
                cov_lagged,
                cov_differenced,
                cov_differenced_lagged,
                cov_lagged_differenced
            ),
        )

        weights = opt_result.x
        adf_stat = opt_result.fun
        B = (weights/weights[0])[1]

        return {'optimal_B': B, 'adf_stat': adf_stat}
    except:
        return {'optimal_B': None, 'adf_stat': None}

def get_weights_0(weights_init):
    """Gets an initial hedge ratio to kickstart the optimisation
    
    Args:
        np.ndarray weights_init: An init if we have any
        
    Returns:
        np.ndarray weights_init: We assume its -1 if no init has weightseen provided
    """
    
    if weights_init is None:
        return [1, -1]
    else:
        return weights_init
    
def get_prices_cov_matrices(prices):
    """
    Create and populate the covariance matrices dictionary.
    
    Args:
        np.ndarray prices: An array of prices, each row
        represents a timestamp, and each column a different
        security
    """
    
    N = prices.shape[1]
    
    prices_differenced = np.diff(prices, axis=0)
    prices_lagged = prices[:-1, :]
    
    ext_cov_differenced_lagged = np.cov(
        np.concatenate([prices_differenced, prices_lagged], axis=1), 
        rowvar=False
    )
    
    cov_p = np.cov(prices, rowvar=False)
    cov_differenced = ext_cov_differenced_lagged[:N, :N]
    cov_lagged = ext_cov_differenced_lagged[N:, N:]
    cov_differenced_lagged = ext_cov_differenced_lagged[:N, N:]
    cov_lagged_differenced = ext_cov_differenced_lagged[N:, :N]
    
    return (
        cov_p,
        cov_differenced,
        cov_lagged,
        cov_differenced_lagged,
        cov_lagged_differenced
    )

def get_DF(weights, prices):
    """
    Provides the Dickey Fuller Direct estimation.
    
    Args:
       np.ndarray weights: Our guess of the optimal hedge ratio
       np.ndarray prices: An array of prices
            
    Returns:
        double DF: The estimated numweightser for Dickey Fuller test statistic.
    """
    
    (
        spread_t,
        spread_differenced,
        spread_lagged,
        T,
        rho
    ) = get_spread(weights, prices)
    
    return float(np.sqrt((T - 2) / (1 - rho ** 2)) * rho)

def get_DF_gradient(weights, 
                    prices, 
                    cov_lagged,
                    cov_differenced,
                    cov_differenced_lagged,
                    cov_lagged_differenced):
    """
    Calculate the Dickey Fuller gradient vector
    
    Args:
        np.ndarray weights: Our guess of the optimal hedge ratio
        np.ndarray prices: An array of prices
        np.ndarray cov_lagged
        np.ndarray cov_differenced
        np.ndarray cov_lagged_differenced
        np.ndarray cov_differenced_lagged
            
    Returns:
        numpy.ndarray : 1D array with the gradient with respect to weights
    """
    
    (
        spread_t,
        spread_differenced,
        spread_lagged,
        T,
        rho
    ) = get_spread(weights, prices)

    sigma_spread_differenced = np.std(spread_differenced)
    sigma_spread_lagged = np.std(spread_lagged)
    
    rho_gradient = get_rho_gradient(
        weights,
        cov_lagged,
        cov_differenced,
        cov_differenced_lagged,
        cov_lagged_differenced,
        sigma_spread_differenced,
        sigma_spread_lagged,
        rho
    )
    
    return (
        np.sqrt((T - 2) / (1 - rho ** 2))
        * (1 + (rho ** 2) / (1 - rho ** 2))
        * rho_gradient
    )

def get_spread(weights, prices):
    """
    Gets the spread, its differenced and lagged series,
    and its correlation coefficient weightsetween the lagged and differenced series.
    
    Args:
        np.ndarray weights: Our guess of the optimal hedge ratio
        np.ndarray prices: An array of prices
    """
    
    spread_t = get_spread_t(weights, prices)
    spread_differenced = np.diff(spread_t)
    spread_lagged = spread_t[:-1]

    T = len(spread_t) - 1
    rho = np.corrcoef(spread_lagged, spread_differenced)[0, 1]
    
    return (
        spread_t,
        spread_differenced,
        spread_lagged,
        T,
        rho
    )

def get_spread_t(weights, prices):
    """
    Gets the spread series
    
    Args:
        np.ndarray weights: Our guess of the optimal hedge ratio
        np.ndarray prices: An array of prices
        
    Returns:
        np.ndarray spread_series
    """
    
    return np.dot(weights, prices.T)

def get_rho_gradient(weights,
                     cov_lagged,
                     cov_differenced,
                     cov_differenced_lagged,
                     cov_lagged_differenced,
                     sigma_spread_differenced,
                     sigma_spread_lagged,
                     rho):
    """
    Calculate the gradient vector of rho (the correlation of spread_differenced and spread_lagged).
    
    Args:
        np.ndarray weights: Our guess of the optimal hedge ratio
        np.ndarray cov_lagged
        np.ndarray cov_differenced
        np.ndarray cov_lagged_differenced
        np.ndarray cov_differenced_lagged
        double sigma_spread_differenced
        double sigma_spread_lagged
        double rho: Correlation of spread_differenced and spread_lagged at weights
            
    Returns:
        numpy.ndarray rho_gradient: 1D array with the gradient with respect to weights
    """
    
    term_1 = np.matmul(
        cov_lagged_differenced 
        +cov_differenced_lagged, 
        weights
    ) / (sigma_spread_lagged * sigma_spread_differenced)
    
    term_2 = -rho * np.matmul(
        cov_lagged / sigma_spread_lagged ** 2
        + cov_differenced / sigma_spread_differenced ** 2,
        weights,
    )
    
    return term_1 + term_2

def get_hessian_DF(weights, 
                   prices, 
                   cov_lagged,
                   cov_differenced,
                   cov_differenced_lagged,
                   cov_lagged_differenced
                  ):
    """
    Calculate the Dickey Fuller Hessian matrix.
    
    Args:
        np.ndarray weights: Our guess of the optimal hedge ratio
        np.ndarray prices: An array of prices
        np.ndarray cov_lagged
        np.ndarray cov_differenced
        np.ndarray cov_lagged_differenced
        np.ndarray cov_differenced_lagged
    
    Returns:
        2D numpy hessian_DF: array with the Hessian with respect to weights
    """
    
    (
        spread_t,
        spread_differenced,
        spread_lagged,
        T,
        rho
    ) = get_spread(weights, prices)
    
    sigma_spread_differenced = np.std(spread_differenced)
    sigma_spread_lagged = np.std(spread_lagged)
    
    grad_rho = get_rho_gradient(
        weights,
        cov_lagged,
        cov_differenced,
        cov_differenced_lagged,
        cov_lagged_differenced,
        sigma_spread_differenced,
        sigma_spread_lagged,
        rho
    )

    hessian_rho = get_hessian_rho(
        weights,
        cov_lagged,
        cov_differenced,
        cov_differenced_lagged,
        cov_lagged_differenced,
        sigma_spread_differenced,
        sigma_spread_lagged,
        rho
    )
    
    return np.sqrt((T - 2) / (1 - rho ** 2) ** 3) * (
        hessian_rho
        + (3 * rho / (1 - rho ** 2)) * np.outer(grad_rho.T, grad_rho)
    )

def get_hessian_rho(weights,
                    cov_lagged,
                    cov_differenced,
                    cov_differenced_lagged,
                    cov_lagged_differenced,
                    sigma_spread_differenced,
                    sigma_spread_lagged,
                    rho):
    """
    Calculate the Hessian matrix of rho (the correlation of spread_differenced and spread_lagged)
    
    Args:
        np.ndarray weights: Our guess of the optimal hedge ratio
        np.ndarray prices: An array of prices
        np.ndarray cov_lagged
        np.ndarray cov_differenced
        np.ndarray cov_lagged_differenced
        np.ndarray cov_differenced_lagged
        double sigma_spread_differenced
        double sigma_spread_lagged
            
    Returns:
        numpy.ndarray hessian_rho: 2D numpy array with the Hessian with respect to weights
    """
    
    G = cov_lagged / (sigma_spread_lagged ** 2) + cov_differenced / (sigma_spread_differenced ** 2)
    K = cov_differenced_lagged + cov_lagged_differenced
    Q_4tensor = (
        np.einsum("ng,ma -> ngma", cov_lagged, cov_lagged) / sigma_spread_lagged ** 4
        + np.einsum("ng,ma -> ngma", cov_differenced, cov_differenced) / sigma_spread_differenced ** 4
    )
    term_1 = (1.0 / (sigma_spread_lagged * sigma_spread_differenced)) * (
        K
        - np.einsum("n, m, ma, ng -> ga", weights, weights, K, G)
        - np.einsum("n, m, ma, ng -> ga", weights, weights, G, K)
    )
    term_2 = rho * (
        np.einsum("n, m, ng, ma -> ga", weights, weights, G, G)
        + 2 * np.einsum("n, m, ngma -> ga", weights, weights, Q_4tensor)
        - G
    )
    
    return term_1 + term_2