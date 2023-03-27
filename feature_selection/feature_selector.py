import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, matthews_corrcoef
from timeseriescv.cross_validation import CombPurgedKFoldCV

class FeatureSelector:
    def __init__(self,
                 features_df,
                 label_series,
                 impact_series,
                 label_type,
                 events_df,
                 returns_df,
                 feature_mean_threshold=0.,
                 feature_noise_z_stat_threshold=1,
                 bar_size=pd.Timedelta('1D'),
                 rf_uniqueness_modifier=None,
                 rf_n_estimators=100,
                 rf_n_jobs=-1,
                 cpcv_n_splits=10,
                 cpcv_n_test_splits=1,
                 cpcv_embargo_td=pd.Timedelta('365D'),
                 verbose=False,
                 ):
        
        self.features_df = features_df
        self.label_series = label_series
        self.impact_series = impact_series
        self.label_type = label_type
        self.events_df = events_df
        self.returns_df = returns_df

        self.feature_mean_threshold = feature_mean_threshold
        self.feature_noise_z_stat_threshold = feature_noise_z_stat_threshold

        self.bar_size = bar_size
        self.rf_uniqueness_modifier = rf_uniqueness_modifier
        self.rf_n_estimators = rf_n_estimators
        self.rf_n_jobs = rf_n_jobs

        self.verbose = verbose

        self.cpcv = CombPurgedKFoldCV(
            n_splits=cpcv_n_splits,
            n_test_splits=cpcv_n_test_splits,
            embargo_td=cpcv_embargo_td
        )
        self.cpcv_splits = list(self.cpcv.split(
            X=self.features_df,
            y=self.label_series,
            pred_times=self.events_df['prediction_date'],
            eval_times=self.events_df['realization_date']
        ))

        self.uniqueness = self.get_average_uniqueness(
            events_df=self.events_df,
            returns_df=self.returns_df,
            bar_size=self.bar_size,
            uniqueness_modifier=self.rf_uniqueness_modifier,
        )

        tree_sample_size = (
            self.uniqueness
            *len(self.events_df)
            *((cpcv_n_splits-cpcv_n_test_splits)/cpcv_n_splits)
        )
        logger.info(f'\n\n---GENERATING MODELS WITH UNIQUENESS: {round(self.uniqueness, 5)} | TREE SAMPLE SIZE: {round(tree_sample_size, 5)}---')

        if label_type == 'classification':
            self.feature_rf = RandomForestClassifier(
                n_estimators=self.rf_n_estimators,
                max_samples=self.uniqueness,
                class_weight='balanced_subsample',
                n_jobs=self.rf_n_jobs,
            )
            self.noise_rf = RandomForestClassifier(
                n_estimators=100,
                max_samples=self.uniqueness,
                class_weight='balanced_subsample',
                n_jobs=self.rf_n_jobs,
            )
        elif label_type == 'regression':
            self.feature_rf = RandomForestRegressor(
                n_estimators=self.rf_n_estimators,
                max_samples=self.uniqueness,
                n_jobs=self.rf_n_jobs,
            )
            self.noise_rf = RandomForestRegressor(
                n_estimators=100,
                max_samples=self.uniqueness,
                n_jobs=self.rf_n_jobs,
            )

    def get_walk_forward_features(self, n_feature_subset=1):
        selected_features = set([])
        selected_features_count = []
        for feature_subset in tqdm(combinations(self.features_df.columns, n_feature_subset)):
            feature_subset = list(feature_subset)
            if self.verbose:
                logger.info(f'\n\n---RUNNING CV FOR FEATURE SUBSET: {feature_subset}---')

            feature_score_cv = []
            noise_score_cv = []
            for train_indices, test_indices in self.cpcv_splits:
                y_train = self.label_series.loc[train_indices].values
                y_test = self.label_series.loc[test_indices].values

                # getting feature Xs
                X_feature_train = self.features_df.loc[train_indices, feature_subset].values
                X_feature_test = self.features_df.loc[test_indices, feature_subset].values

                X_noise_train = np.stack([
                    np.random.normal(
                        np.mean(self.features_df.loc[train_indices, feature]),
                        np.std(self.features_df.loc[train_indices, feature]),
                        self.features_df.loc[train_indices, feature].shape
                    )
                    for feature in feature_subset
                ], axis=0).T
                X_noise_test = np.stack([
                    np.random.normal(
                        np.mean(self.features_df.loc[test_indices, feature]),
                        np.std(self.features_df.loc[test_indices, feature]),
                        self.features_df.loc[test_indices, feature].shape
                    )
                    for feature in feature_subset
                ], axis=0).T

                if len(feature_subset) == 1:
                    X_feature_train = X_feature_train.reshape(-1, 1)
                    X_feature_test = X_feature_test.reshape(-1, 1)
                    X_noise_train = X_noise_train.reshape(-1, 1)
                    X_noise_test = X_noise_test.reshape(-1, 1)

                impact_weights = self.impact_series.loc[train_indices]
                impact_weights = impact_weights.abs() / impact_weights.abs().sum()

                # training the rf models
                self.feature_rf.fit(
                    X_feature_train,
                    y_train,
                    sample_weight=impact_weights.values,
                )
                self.noise_rf.fit(
                    X_noise_train,
                    y_train
                )

                if self.label_type == 'classification':
                    feature_score = matthews_corrcoef(
                        y_test,
                        self.feature_rf.predict(X_feature_test)
                    )
                    noise_score = matthews_corrcoef(
                        y_test,
                        self.noise_rf.predict(X_noise_test)
                    )
                elif self.label_type == 'regression':
                    feature_score = r2_score(
                        y_test,
                        self.feature_rf.predict(X_feature_test)
                    )
                    noise_score = r2_score(
                        y_test,
                        self.noise_rf.predict(X_noise_test)
                    )
                feature_score_cv.append(feature_score)
                noise_score_cv.append(noise_score)

            feature_mean = np.mean(feature_score_cv)
            feature_std = np.std(feature_score_cv)
            noise_mean = np.mean(noise_score_cv)
            noise_std = np.std(noise_score_cv)
            z_stat = (
                (feature_mean-noise_mean)
                /np.sqrt(feature_std**2 + noise_std**2)
            )

            if self.verbose:
                logger.info(f'\n---FEATURE SUBSET: {feature_subset} MEAN: {round(feature_mean, 5)}---')
                logger.info(f'\n---FEATURE SUBSET: {feature_subset} Z STAT: {round(z_stat, 5)}---')

            # feature subset isn't noise
            if ((feature_mean >= self.feature_mean_threshold)
            and (z_stat >= self.feature_noise_z_stat_threshold)):
                if self.verbose:
                    logger.info(f'---FEATURE SUBSET ADDED: {feature_subset}---')
                selected_features = selected_features.union(
                    set(feature_subset)
                )
                selected_features_count.extend(feature_subset)
            else:
                if self.verbose:
                    logger.info(f'---FEATURE SUBSET NO PREDICTIVE POWER: {feature_subset}---')

        return selected_features, selected_features_count

    def get_average_uniqueness(self,
                               events_df,
                               returns_df,
                               bar_size,
                               uniqueness_modifier=None):
        events_df['realization_date'] = (
            events_df['realization_date'].fillna(returns_df.index[-1])
        )
        events_df = events_df.sort_values(by='prediction_date')

        first_observation_time = events_df['prediction_date'].iloc[0]
        last_observation_time = events_df['realization_date'].max()
        returns_df = returns_df.loc[
            (first_observation_time <= returns_df.index)
            & (returns_df.index <= last_observation_time)
        ]

        timeframes_df = events_df[['prediction_date', 'realization_date']]
        timeframes_df = (timeframes_df - first_observation_time) // bar_size

        events_array = timeframes_df.values

        concurrent_events = dict()
        for label in range(events_array.shape[0]):
            start, end = events_array[label]
            for t in range(start, end):
                concurrent_events[t] = concurrent_events.get(t, 0)+1

        samples = events_array.shape[0]
        avg_uniqueness = np.zeros(samples)

        for label in range(samples):
            start, end = events_array[label]
            for t in range(start, end):
                avg_uniqueness[label] += 1/concurrent_events[t]
            avg_uniqueness[label] /= (end-start)

        average_uniqueness = np.mean(avg_uniqueness)

        if uniqueness_modifier == 'sqrt':
            average_uniqueness = np.sqrt(average_uniqueness)
        elif isinstance(uniqueness_modifier, float):
            average_uniqueness = uniqueness_modifier

        return average_uniqueness