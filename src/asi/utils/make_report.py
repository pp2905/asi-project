import pandas as pd
import pandas_profiling


def make_report():
    data = pd.read_csv("data/processed/dataset1_processed.csv")
    profile = data.profile_report()
    profile.to_file("data/data_profile_report.html")


make_report()
