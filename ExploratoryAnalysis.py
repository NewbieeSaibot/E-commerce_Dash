import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ExploratoryAnalysisLittleData:
    def __init__(self):
        pass

    @staticmethod
    def look_columns(data: pd.DataFrame) -> None:
        for col in data.columns:
            print(col, data[col].dtype)

        print(data.describe())

    @staticmethod
    def first_lines(data: pd.DataFrame) -> None:
        print(data.head())

    @staticmethod
    def sells_per_year(data: pd.DataFrame) -> None:
        sum_by_year = data.groupby("year").sum()
        print(sum_by_year["gross_revenue"])
        print(sum_by_year["net_revenue"])
        print(sum_by_year["boxes"])

    @staticmethod
    def mean_ticket_per_acquisition_channel_per_year(data: pd.DataFrame) -> None:
        for year in data["year"].unique():
            print(f"Ano de {year}")
            data_year = data[data["year"] == year]
            mean_per_acquisition_channel = data_year.groupby("customer_acquisition_channel").mean()
            print("Ticket médio por acquisition channel:", mean_per_acquisition_channel["gross_revenue"])

    @staticmethod
    def unique_customers_per_year(data: pd.DataFrame) -> None:
        for year in data["year"].unique():
            print(f"Ano de {year}")
            data_year = data[data["year"] == year]
            print(len(data_year['customer_id'].unique()))

    @staticmethod
    def total_sales_for_each_client_histogram(data: pd.DataFrame) -> None:
        sales_quantity_per_client = data.groupby("customer_id").count()["week"].values
        plt.hist(sales_quantity_per_client, bins=max(sales_quantity_per_client) - 1)
        plt.show()

    @staticmethod
    def probability_to_buy_based_on_previous_sells(data: pd.DataFrame):
        sales_quantity_per_client = data.groupby("customer_id").count()["week"]
        aux = pd.DataFrame()
        aux['sales_quantity_per_client'] = sales_quantity_per_client
        aux['id'] = np.arange(len(aux))
        aux_dic = aux.groupby("sales_quantity_per_client").count()['id'].to_dict()
        chaves = list(aux_dic.keys())
        for k1 in range(1, max(list(aux_dic.keys())) + 1):
            for k2 in chaves:
                if k2 > k1:
                    try:
                        aux_dic[k1] += aux_dic[k2]
                    except:
                        aux_dic[k1] = aux_dic[k2]

        das = pd.DataFrame()
        das['sales_quantity'] = np.arange(1, 153)
        ar_a = np.zeros(152)
        for i in range(1, 153):
            ar_a[i-1] = aux_dic[i]

        das['times'] = ar_a
        print(ar_a)
        ar_b = np.zeros(len(ar_a))
        for i in range(1, len(ar_a)):
            ar_b[i] = ar_a[i] / ar_a[i - 1]
        das['Prob'] = ar_b

        fig, ax = plt.subplots()
        ax.plot(das['times'])
        ax.set_xlabel("Ordem da venda")
        ax.set_ylabel("Quantidade de vezes")

        plt.show()

        fig, ax = plt.subplots()
        ax.plot(das['Prob'].iloc[1:20])
        ax.set_title("Probabilidade do cliente voltar a comprar no E-commerce baseado em número de compras anteriores.")
        ax.set_xlabel("Cliente que fez N compras.")
        ax.set_ylabel("Probabilidade de voltar a comprar pela vez N + 1.")

        plt.show()

    @staticmethod
    def customer_lifetime_value(data: pd.DataFrame) -> None:
        customer_lifetime_value = data.groupby("customer_id").sum()["gross_revenue"].values
        print(np.percentile(customer_lifetime_value, [5, 25, 50, 75, 95]))
        plt.hist(customer_lifetime_value, bins=200)
        plt.show()

    @staticmethod
    def simple_plot(data: pd.DataFrame) -> None:
        # Calculate active clients
        alpha = 0.9718
        active_clients = np.zeros(len(data))
        for i in range(len(data)):
            # print(i)
            for j in range(i, -1, -1):
                # print(f"iter {j} exp {i - j}")
                active_clients[i] += data["unique_clients_week"].iloc[j] * alpha ** (i - j)

        data['active_clients'] = active_clients
        print(alpha, data[['active_clients', 'net_revenue']].corr())

        for year in [2013, 2014, 2015]:
            plot_data = data[data['year'] == year]
            fig, ax = plt.subplots()
            ax.plot(plot_data["week"], plot_data["net_revenue"], color="red", label=f"{year}")

            ax.set_title(f"{year}")
            ax.legend()
            plt.show()

        for year in [2013, 2014, 2015]:
            plot_data = data[data['year'] == year]
            fig, ax = plt.subplots()
            ax.plot(plot_data["week"], plot_data["unique_clients_week"], color="red", label=f"{year}")

            ax.set_title(f"{year}")
            ax.legend()
            plt.show()

        for year in [2013, 2014, 2015]:
            plot_data = data[data['year'] == year]
            fig, ax = plt.subplots()
            ax.plot(plot_data["week"], plot_data["active_clients"], color="red", label=f"{year}")
            ax.plot(plot_data["week"], plot_data["net_revenue"]/10, color="black", label=f"{year}")

            ax.set_title(f"{year}")
            ax.legend()
            plt.show()


if __name__ == "__main__":
    df = pd.read_csv("./data/Dataset_teste_Just_BI.csv", sep=";")
    df2 = pd.read_csv("./data/dataset_preprocessed_1.csv")

    exp_analysis = ExploratoryAnalysisLittleData()
    # exp_analysis.look_columns(df)
    # exp_analysis.first_lines(df)
    # exp_analysis.sells_per_year(df)
    # exp_analysis.mean_ticket_per_acquisition_channel_per_year(df)
    # exp_analysis.unique_customers_per_year(df)
    # Extra information about sales
    # exp_analysis.total_sales_for_each_client_histogram(df)
    # exp_analysis.probability_to_buy_based_on_previous_sells(df)
    # exp_analysis.customer_lifetime_value(df)
    exp_analysis.simple_plot(df2)
