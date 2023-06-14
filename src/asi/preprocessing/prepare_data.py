import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from asi.utils.helpers import object_to_int
import wandb


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

    wandb.init(
        project="my-awesome-project", entity="asi-project-2023", tags=["Dataset"]
    )
    # Create a new artifact for the file repository
    file_artifact = wandb.Artifact("raw_file", type="dataset", description="Raw file")
    # Add files to the artifact
    file_artifact.add_file(input_file)

    # Log the artifact
    wandb.log_artifact(file_artifact)

    # Link the artifact to a specific location in your project
    wandb.run.link_artifact(
        file_artifact, "asi-project-2023/my-awesome-project/Dataset"
    )

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    mean_total_charges = df["TotalCharges"].mean()
    df["TotalCharges"] = df["TotalCharges"].fillna(mean_total_charges)
    df.drop("customerID", axis=1, inplace=True)

    encode_categorical(df, CATEGORICAL_COLUMN)
    normalize_numeric(df, NUMERIC_COLUMNS)

    df = df.apply(lambda x: object_to_int(x))
    df.to_csv(output_file, index=False)

    file_artifact = wandb.Artifact(
        "processed_file", type="dataset", description="Processed File"
    )
    # Add files to the artifact
    file_artifact.add_file(output_file)

    # Log the artifact
    wandb.log_artifact(file_artifact)

    # Link the artifact to a specific location in your project
    wandb.run.link_artifact(
        file_artifact, "asi-project-2023/my-awesome-project/Dataset"
    )

    wandb.finish()


def encode_categorical(df, columns):
    df_encoded = pd.get_dummies(df, columns=columns, drop_first=True)
    return df_encoded


def normalize_numeric(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df
