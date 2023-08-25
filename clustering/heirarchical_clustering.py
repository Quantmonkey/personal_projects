import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster, cut_tree

from models.covariance.covariance_model import CovarianceModel

class HierarchicalClustering:
    def __init__(self):
        self.CORR_ZERO_TOL = 1e-5
    
    def get_cluster_mappings(self, rets_df, max_cluster_num=20):

        corr_df, distance_df = self.get_distance_df(rets_df)
        flatten_distance_array = squareform(distance_df.values, checks=False)
        linkage_matrix = linkage(
            flatten_distance_array,
            method='ward',
            optimal_ordering=True
        )

        cluster_tags = fcluster(linkage_matrix,
            t=self.get_opt_cluster_num_from_gap_stat(
                corr_df,
                distance_df,
                linkage_matrix,
                max_cluster_num=max_cluster_num
            ),
            criterion='maxclust'
        )

        sec_to_cluster = dict(zip(distance_df.columns, cluster_tags))
        cluster_to_sec = {}
        for security, cluster in sec_to_cluster.items():
            cluster_to_sec[cluster] = (
                cluster_to_sec.get(cluster, []) + [security]
            )

        return sec_to_cluster, cluster_to_sec

    def get_distance_df(self, rets_df):
        (denoised_corr,
        denoised_e_val,
        denoised_e_vec,
        e_max) = CovarianceModel().get_clean_corr_df(rets_df, detone=True)

        denoised_corr = pd.DataFrame(denoised_corr, columns=rets_df.columns, index=rets_df.columns)
        np.fill_diagonal(denoised_corr.values, 1)

        distance_matrix = np.sqrt(np.clip(
            a=(1-denoised_corr)/2,
            a_min=0,
            a_max=1,
        ))
        distance_matrix[distance_matrix < self.CORR_ZERO_TOL] = 0

        return denoised_corr, distance_matrix

    def get_opt_cluster_num_from_gap_stat(self,
                                          corr_df,
                                          dist_df,
                                          linkage_matrix,
                                          max_cluster_num):

        cluster_lvls = pd.DataFrame(
            cut_tree(linkage_matrix),
            index=corr_df.columns,
        )
        num_k = cluster_lvls.columns
        cluster_lvls = cluster_lvls.iloc[:, ::-1]
        cluster_lvls.columns = num_k
        W_list = []

        for k in range(min(len(cluster_lvls.columns), max_cluster_num)):
            level = cluster_lvls.iloc[:, k]
            D_list = []

            for i in range(np.max(level.unique()) + 1):
                cluster = level.loc[level == i]
                cluster_dist_df = dist_df.loc[cluster.index, cluster.index]
                cluster_pdist_df = squareform(cluster_dist_df, checks=False)

                if cluster_pdist_df.shape[0] != 0:
                    D = np.nan_to_num(cluster_pdist_df.mean())
                    D_list.append(D)

            W_k = np.sum(D_list)
            W_list.append(W_k)

        W_list = pd.Series(W_list)
        n = corr_df.shape[0]
        limit_k = int(min(max_cluster_num, np.sqrt(n)))
        gaps = W_list.shift(2) + W_list - 2 * W_list.shift(1)
        gaps = gaps[:limit_k]
        if gaps.isna().all():
            opt_cluster_num = len(gaps)
        else:
            opt_cluster_num = int(gaps.idxmax()+2)

        return opt_cluster_num
