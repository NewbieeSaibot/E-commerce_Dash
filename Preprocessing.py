import pandas as pd


class PreprocessingLittleData:
    def __init__(self):
        pass

    @staticmethod
    def get_kpis_per_week_dataset(raw_data: pd.DataFrame) -> pd.DataFrame:
        # Get net revenue in function of time // year, week
        def convert_week_to_numeric(value):
            return int(str(value).split("W")[1])

        raw_data['week'] = raw_data['week'].map(convert_week_to_numeric)
        raw_data.sort_values(by="week", ascending=True, kind="mergesort", inplace=True)
        raw_data.sort_values(by="year", ascending=True, kind="mergesort", inplace=True)

        print(raw_data)

        year, week = [], []
        net_revenue = []
        unique_clients_week_number = []
        gross_revenue = []
        boxes = []
        unique_clients = set()
        for y in raw_data['year'].unique():
            print(f"year: {y}")
            year_data = raw_data[raw_data['year'] == y]
            net_revenue_for_each_week = year_data.groupby("week").sum()['net_revenue']
            gross_revenue_for_each_week = year_data.groupby("week").sum()['gross_revenue']
            boxes_for_each_week = year_data.groupby("week").sum()['boxes']

            for wk in net_revenue_for_each_week.index:
                print(f"week: {wk}")
                week.append(wk)
                year.append(y)
                ini = len(unique_clients)
                print("unique clients len:", len(unique_clients))
                unique_clients = unique_clients.union(set(year_data[year_data['week'] == wk]['customer_id'].unique()))
                unique_clients_week_number.append(len(unique_clients) - ini)

            for nr in net_revenue_for_each_week.values:
                net_revenue.append(nr)

            for gr in gross_revenue_for_each_week.values:
                gross_revenue.append(gr)

            for bx in boxes_for_each_week.values:
                boxes.append(bx)

        data = pd.DataFrame()
        data['year'] = year
        data['week'] = week
        data['net_revenue'] = net_revenue
        data['unique_clients_week'] = unique_clients_week_number
        data['gross_revenue'] = gross_revenue
        data['boxes'] = boxes
        return data

    @staticmethod
    def get_kpis_per_year_dataset(raw_data: pd.DataFrame) -> pd.DataFrame:
        # Get net revenue in function of time // year, week
        def convert_week_to_numeric(value):
            return int(str(value).split("W")[1])

        raw_data['week'] = raw_data['week'].map(convert_week_to_numeric)
        raw_data.sort_values(by="week", ascending=True, kind="mergesort", inplace=True)
        raw_data.sort_values(by="year", ascending=True, kind="mergesort", inplace=True)

        data = raw_data.groupby("year").sum()[['gross_revenue', 'net_revenue', 'boxes']]
        kpi = []
        for year in [2013, 2014, 2015]:
            kpi.append(len(raw_data[raw_data['year'] == year]['customer_id'].unique()))
        data['unique_clients_per_year'] = kpi

        kpi = []
        for year in [2013, 2014, 2015]:
            kpi.append(raw_data[raw_data['year'] == year][raw_data['customer_acquisition_channel'] == "Paid Marketing"]
                       ['gross_revenue'].mean())
        data['mean_ticket_paid_marketing'] = kpi

        kpi = []
        for year in [2013, 2014, 2015]:
            kpi.append(raw_data[raw_data['year'] == year][raw_data['customer_acquisition_channel'] == "Referral"]
                       ['gross_revenue'].mean())
        data['mean_ticket_referral'] = kpi

        return data


if __name__ == "__main__":
    df = pd.read_csv("./data/Dataset_teste_Just_BI.csv", sep=";")

    preprocessor = PreprocessingLittleData()
    # dat = preprocessor.get_kpis_per_week_dataset(df)
    dat = preprocessor.get_kpis_per_year_dataset(df)
    dat.to_csv("./data/dataset_preprocessed_2.csv", index=False)
