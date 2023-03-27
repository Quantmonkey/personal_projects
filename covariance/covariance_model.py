import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut

class CovarianceModel:
    def __init__(self):
        pass

    def get_clean_corr_df(self, rets_df, detone=False):
        q = rets_df.shape[0]/rets_df.shape[1]
        corr_df = rets_df.corr().dropna(axis=1, how='all').dropna(how='all')

        e_val, e_vec = self.get_pca(corr_df)
        e_max, var = self.find_max_eigen_val(np.diag(e_val), q, self.find_optimal_bandwidth(np.diag(e_val)))

        denoised_corr = self.denoise_corr(e_val, e_vec, e_max)
        denoised_e_val, denoised_e_vec = self.get_pca(denoised_corr)

        if detone:
            denoised_corr = self.detone_corr(denoised_e_val, denoised_e_vec, n_facts=1)
            denoised_e_val, denoised_e_vec = self.get_pca(denoised_corr)

        return denoised_corr, denoised_e_val, denoised_e_vec, e_max

    def get_clean_cov_df(self, rets_df, detone=False):

        cov_df = rets_df.cov()
        cov_df = cov_df.loc[~(cov_df==0).all(axis=1)]
        cov_df = cov_df.loc[:, (cov_df!=0).any(axis=0)]

        std = np.sqrt(np.diag(cov_df))

        (denoised_corr, 
        denoised_e_val, 
        denoised_e_vec, 
        e_max) = self.get_clean_corr_df(rets_df=rets_df, detone=detone)

        denoised_cov = self.corr2cov(denoised_corr, std)

        return denoised_cov, denoised_e_val, denoised_e_vec, e_max

    def mp_pdf(self, var, q, pts):

        e_min = var*(1 - (1./q)**0.5)**2
        e_max = var*(1 + (1./q)**0.5)**2
        e_val = np.linspace(e_min, e_max, pts)
        pdf = q*((e_max-e_val)*(e_val-e_min))**0.5/(2*np.pi*var*e_val)

        return pd.Series(pdf, index=e_val)

    def get_pca(self, matrix):
        e_val, e_vec = np.linalg.eigh(matrix)
        indices = e_val.argsort()[::-1]
        e_val = e_val[indices]
        e_vec = e_vec[:, indices]
        e_val = np.diagflat(e_val)
        return e_val, e_vec
    
    def fit_KDE(self, obs, bwidth=0.25, kernel='gaussian', x=None):
        if len(obs.shape) == 1:
            obs = obs.reshape(-1, 1)
        kde = KernelDensity(kernel=kernel, bandwidth=bwidth).fit(obs)
        if x is None:
            x = np.unique(obs).reshape(-1 , 1)
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        log_prob = kde.score_samples(x)
        pdf = pd.Series(np.exp(log_prob), index=x.flatten())
        return pdf

    def find_optimal_bandwidth(self, e_val):

        bandwidths = 10**np.linspace(-2, 1, 5)
        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths},
                            cv=LeaveOneOut())
        grid.fit(e_val[:, None])

        return grid.best_params_['bandwidth']

    def err_pdf(self, var, e_val, q, bwidth, pts=50):
        pdf0 = self.mp_pdf(var[0], q, pts)
        pdf1 = self.fit_KDE(e_val, bwidth, x=pdf0.index.values)
        sse = np.sum((pdf1 - pdf0) ** 2)
        return sse

    def find_max_eigen_val(self, e_val, q, bwidth, min_var=1e-5, max_var=1-1e-5):
        out = minimize(lambda *x: self.err_pdf(*x), .5, args=(e_val, q, bwidth), bounds=((min_var, max_var),))
        if out["success"]:
            var = out['x'][0]
        else:
            var = 1
        e_max = int(var * (1 + (1./q) ** 0.5) ** 2)
        return e_max, var

    def denoise_corr(self, e_val, e_vec, n_facts, shrinkage=False, alpha=0):
        if shrinkage:
            e_val_l, e_vec_l = e_val[:n_facts, :n_facts], e_vec[:, :n_facts]
            e_val_r, e_vec_r = e_val[n_facts:, n_facts:], e_vec[:, n_facts:]
            corr_l = np.dot(e_vec_l, e_val_l).dot(e_vec_l.T)
            corr_r = np.dot(e_vec_r, e_val_r).dot(e_vec_r.T)
            corr1 = corr_l + alpha * corr_r + (1 - alpha) * np.diag(np.diag(corr_r))
        else:
            e_val_ = np.diag(e_val).copy()
            e_val_[n_facts:] = e_val_[n_facts:].sum() / float(e_val_.shape[0] - n_facts)
            e_val_ = np.diag(e_val_)
            corr1 = np.dot(e_vec, e_val_).dot(e_vec.T)
            # Renormalize to keep trace 1
            corr1 = self.cov2corr(corr1)
        return corr1

    def detone_corr(self, e_val, e_vec, n_facts, shrinkage=False, alpha=0):
        if shrinkage:
            e_val_r, e_vec_r = e_val[n_facts:, n_facts:], e_vec[:, n_facts:]
            corr_r = np.dot(e_vec_r, e_val_r).dot(e_vec_r.T)
            corr1 = alpha * corr_r + (1 - alpha) * np.diag(np.diag(corr_r))
            # Renormalize to keep trace 1
            corr1 = self.cov2corr(corr1)
        else:
            e_val_ = np.diag(e_val).copy()
            e_val_[:n_facts] = 0
            e_val_ = np.diag(e_val_)
            corr1 = np.dot(e_vec, e_val_).dot(e_vec.T)
            # Renormalize to keep trace 1
            corr1 = self.cov2corr(corr1)
        return corr1

    def denoise_cov(self, cov, q, bwidth):
        corr0 = self.cov2corr(cov)
        e_val0, e_vec0 = self.get_pca(corr0)
        e_max0, var0 = self.find_max_eigen_val(np.diag(e_val0), q, bwidth)
        nfacts0 = e_val0.shape[0] - np.diag(e_val0)[::-1].searchsorted(e_max0)
        corr1 = self.denoise_corr(e_val0, e_vec0, nfacts0)
        cov1 = self.corr2cov(corr1, np.diag(cov) ** .5)
        return cov1

    def opt_portfolio(self, cov, mu=None):
        inv = np.linalg.inv(cov)
        ones = np.ones(shape=(inv.shape[0], 1))
        if mu is None:
            mu = ones
        w = np.dot(inv, mu)
        w /= np.dot(ones.T, w)

        return w

    def cov2corr(self, cov):
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        corr[corr < -1] = -1
        corr[corr > 1] = 1

        return corr

    def corr2cov(self, corr, std):
        return corr * np.outer(std, std)