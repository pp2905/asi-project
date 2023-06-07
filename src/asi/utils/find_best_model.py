import h2o


def find_best_model():
    h2o.init()
    data = h2o.import_file("data/processed/dataset1_processed.csv")
    automl = h2o.automl.H2OAutoML(max_models=10)
    automl.train(y="Churn", training_frame=data)
    best_model = automl.leader
    print(f"best_model: {best_model}")


find_best_model()
