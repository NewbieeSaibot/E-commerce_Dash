import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor


class LittleDataModelling:
    def __init__(self, raw_data: pd.DataFrame, model_class, model_name: str,
                 target: str, metrics: list, n_folds: int):
        self.raw_data = raw_data
        self.model_class = model_class
        self.models = []
        self.target = target
        self.all_features = list(raw_data.columns)
        self.all_features.remove(self.target)
        self.metrics = metrics
        self.selected_features = self.all_features
        self.n_folds = n_folds
        self.model_name = model_name

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
            model = self.model_class()
            model.fit(x_train, y_train)
            y_hat = model.predict(x_val)
            for j in range(len(self.metrics)):
                cross_metrics[j] += self.metrics[j](y_val, y_hat)/self.n_folds

        return cross_metrics

    def feature_engineering(self):
        pass

    def feature_selection(self):
        self.selected_features = ["year", "week"]  # "unique_clients_week"

    def hyper_params_tuning(self):
        pass

    def run_pipeline(self):
        self.feature_engineering()
        self.separate_folds()
        self.feature_selection()
        self.save_predictions()
        print(self.get_ts_cross_validation_score())

    def train_with_all_data_and_predict(self) -> pd.DataFrame:
        x_train = self.raw_data[self.selected_features]
        y_train = self.raw_data[self.target]

        model = self.model_class()
        model.fit(x_train, y_train)

        projection = pd.DataFrame()
        projection['week'] = np.arange(1, 53 // 2)
        projection['year'] = np.zeros(len(projection)) + 2016
        projection['projected_net_revenue'] = model.predict(projection[self.selected_features])
        return projection

    def save_predictions(self):
        projection = self.train_with_all_data_and_predict()
        projection.to_csv(f"./data/projections/{self.model_name}.csv", index=False)


if __name__ == "__main__":
    df = pd.read_csv("./data/dataset_preprocessed_1.csv")
    metrics = [mean_absolute_error, mean_absolute_percentage_error, r2_score]
    target = "net_revenue"
    # target = "unique_clients_week"

    modelling = LittleDataModelling(df, model_class=LinearRegression, model_name="linear_regression", target=target, metrics=metrics, n_folds=10)
    modelling.run_pipeline()

    modelling = LittleDataModelling(df, model_class=RandomForestRegressor, model_name="random_forest", target=target, metrics=metrics, n_folds=10)
    modelling.run_pipeline()

    modelling = LittleDataModelling(df, model_class=MLPRegressor, model_name="mlp", target=target, metrics=metrics, n_folds=10)
    modelling.run_pipeline()
