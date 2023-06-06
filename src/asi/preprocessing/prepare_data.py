import pandas as pd

from asi.uttils.helpers import object_to_int


def prepare_data(input_file: str, output_file: str):
    data = pd.read_csv(input_file)
    data["TotalCharges"] = pd.to_numeric(data.TotalCharges, errors="coerce")
    data.dropna(inplace=True)
    data = data.drop(["customerID"], axis=1)
    data.drop(labels=data[data["tenure"] == 0].index, axis=0, inplace=True)
    data["SeniorCitizen"] = data["SeniorCitizen"].map({0: "No", 1: "Yes"})
    data = data.apply(lambda x: object_to_int(x))
    data.to_csv(output_file, index=False)