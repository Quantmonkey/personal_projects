import sys
sys.path.append(fr"C:\Users\Oscar\git\transparent_offchain\main")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from models.covariance.covariance_model import CovarianceModel

class PCAEigenFactors:
    def __init__(self):
        pass

    def get_sec_to_residual_returns(self, rets_df):
        (eigenfactor_df, 
        factor_importances) = self.get_eigenfactors_from_rets(rets_df)

        sec_to_factor_exposure = {}
        sec_to_residual_returns = {}
        sec_to_beta_hedged_returns = {}
        for sec in rets_df:
            lr = LinearRegression()
            X = eigenfactor_df.values
            y = rets_df[sec].values
            lr.fit(X, y)
            sec_to_factor_exposure[sec] = lr.coef_

            sec_to_beta_hedged_returns[sec] = (
                rets_df[sec].iloc[-1]
                -(lr.coef_[0]*eigenfactor_df.iloc[-1].values[0])
            )

            sec_to_residual_returns[sec] = (
                rets_df[sec].iloc[-1]
                -np.sum(
                    lr.coef_
                    *eigenfactor_df.iloc[-1].values
                )
            )

        return (
            sec_to_residual_returns,
            sec_to_beta_hedged_returns,
            eigenfactor_df,
            sec_to_factor_exposure,
            factor_importances
        )

    def get_sec_to_eigenfactor_exposures(self, rets_df, equal_importance=False):

        eigenfactor_df, factor_importances = self.get_eigenfactors_from_rets(
            rets_df=rets_df,
            equal_importance=equal_importance,
        )

        sec_to_factor_exposure = {}
        for sec in rets_df:
            lr = LinearRegression()
            X = eigenfactor_df.values
            y = rets_df[sec].values
            lr.fit(X, y)
            sec_to_factor_exposure[sec] = lr.coef_

        return sec_to_factor_exposure, factor_importances

    def get_eigenfactors_from_rets(self, rets_df, equal_importance=False):
        (denoised_corr,
        denoised_e_val,
        denoised_e_vec,
        e_max) = CovarianceModel().get_clean_corr_df(rets_df)

        factor_loadings = denoised_e_vec.T[:e_max]

        if equal_importance:
            factor_importances = (1/e_max)*np.ones(e_max)
        else:
            tmp_factor_importance = np.diag(denoised_e_val.T)[:e_max]
            factor_importances = np.array([
                fctr_impt/np.sum(np.abs(tmp_factor_importance))
                for fctr_impt in tmp_factor_importance
            ])

        eigenfactor_wghts_df = pd.DataFrame(factor_loadings).T.apply(
            lambda x: (x/np.sum(np.abs(x)))
        )

        eigenfactor_to_series = {}
        for eigenfactor_idx in eigenfactor_wghts_df:
            eigenfactor_wghts = eigenfactor_wghts_df[eigenfactor_idx]
            eigenfactor_to_series[eigenfactor_idx] = (
                eigenfactor_wghts.values
                *rets_df
            ).sum(axis=1)

        eigenfactor_df = pd.DataFrame(eigenfactor_to_series)

        return eigenfactor_df, factor_importances