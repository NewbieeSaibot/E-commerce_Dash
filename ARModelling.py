import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima.arima import ADFTest, auto_arima
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    time = np.arange(len(data))
    data.sort_values(by="week", ascending=True, kind="mergesort", inplace=True)
    data.sort_values(by="year", ascending=True, kind="mergesort", inplace=True)

    data['time'] = time
    data.set_index("time", inplace=True)
    data = data[['net_revenue']]
    return data


def observing_pattern(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    ax.plot(data["net_revenue"])
    plt.show()


def test_stationarity(data: pd.DataFrame) -> None:
    adf_test = ADFTest(alpha=0.05)
    print(adf_test.should_diff(data))


def main():
    data = pd.read_csv("./data/dataset_preprocessed_1.csv")
    data = preprocess(data)
    observing_pattern(data)
    test_stationarity(data)
    train, test = data.iloc[0:int(len(data)*0.7)], data.iloc[int(len(data)*0.7):]

    arima_model = auto_arima(train)
    print(arima_model.summary())
    prediction = arima_model.predict(n_periods=len(test))

    fig, ax = plt.subplots()
    ax.plot(test.values[0:len(prediction)])
    ax.plot(prediction)
    plt.show()
    print(mean_absolute_error(test.values[0:len(prediction)], prediction))
    print(mean_absolute_percentage_error(test.values[0:len(prediction)], prediction))
    print(r2_score(test.values[0:len(prediction)], prediction))
    arima_model = auto_arima(data)
    prediction = arima_model.predict(n_periods=25)
    projection = pd.DataFrame()
    projection['week'] = np.arange(1, 26)
    projection['year'] = np.zeros(25)
    projection['projected_net_revenue'] = prediction
    projection.to_csv("./data/projections/arima.csv", index=False)


if __name__ == "__main__":
    main()
