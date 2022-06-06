import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from joblib import dump


class LittleDataModelling:
    def __init__(self, raw_data: pd.DataFrame, model, target: str, metrics: list, n_folds: int):
        self.raw_data = raw_data
        self.model = model()
        self.target = target
        self.all_features = list(raw_data.columns)
        self.all_features.remove(self.target)
        self.metrics = metrics
        self.selected_features = self.all_features
        self.n_folds = n_folds

    def separate_folds(self) -> None:
        # Separa os dados em folds
        folds = np.zeros(len(self.raw_data))
        for i in range(self.n_folds):
            folds[int(i*len(self.raw_data)/self.n_folds):] += 1
        self.raw_data['fold'] = folds

    def get_ts_cross_validation_score(self):
        cross_metrics = {}
        for i in range(len(self.metrics)):
            cross_metrics[i] = 0

        for i in range(self.n_folds // 2, self.n_folds):
            x_train = self.raw_data[self.raw_data['fold'] <= i][self.selected_features]
            y_train = self.raw_data[self.raw_data['fold'] <= i][self.target]
            x_val = self.raw_data[self.raw_data['fold'] > i][self.selected_features]
            y_val = self.raw_data[self.raw_data['fold'] > i][self.target]

            self.model.fit(x_train, y_train)
            y_hat = self.model.predict(x_val)
            for j in range(len(self.metrics)):
                cross_metrics[j] += self.metrics[j](y_val, y_hat)/self.n_folds

        return cross_metrics

    def feature_engineering(self):
        pass

    def feature_selection(self):
        self.selected_features = ["year", "week"]  #, "unique_clients_week"]

    def hyper_params_tuning(self):
        pass

    def run_pipeline(self):
        self.feature_engineering()
        self.separate_folds()
        self.feature_selection()
        print(self.get_ts_cross_validation_score())


if __name__ == "__main__":
    df = pd.read_csv("./data/dataset_preprocessed_1.csv")
    metrics = [mean_absolute_error, mean_absolute_percentage_error, r2_score]
    target = "net_revenue"
    # target = "unique_clients_week"

    modelling = LittleDataModelling(df, model=LinearRegression, target=target, metrics=metrics, n_folds=10)
    modelling.run_pipeline()

    modelling = LittleDataModelling(df, model=RandomForestRegressor, target=target, metrics=metrics, n_folds=10)
    modelling.run_pipeline()

    modelling = LittleDataModelling(df, model=MLPRegressor, target=target, metrics=metrics, n_folds=10)
    modelling.run_pipeline()
