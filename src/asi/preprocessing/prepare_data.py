import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from asi.utils.helpers import object_to_int


CATEGORICAL_COLUMN = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


def prepare_data(input_file: str, output_file: str):
    df = pd.read_csv(input_file)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    mean_total_charges = df["TotalCharges"].mean()
    df["TotalCharges"] = df["TotalCharges"].fillna(mean_total_charges)

    encode_categorical(df, CATEGORICAL_COLUMN)
    normalize_numeric(df, NUMERIC_COLUMNS)

    df = df.apply(lambda x: object_to_int(x))
    df.to_csv(output_file, index=False)


def encode_categorical(df, columns):
    df_encoded = pd.get_dummies(df, columns=columns, drop_first=True)
    return df_encoded


def normalize_numeric(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df
