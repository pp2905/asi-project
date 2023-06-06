from __future__ import annotations

from kedro.pipeline import Pipeline, node

from asi.preprocessing.prepare_data import prepare_data
from asi.training.train_model import train_model


def create_prepare_data_pipeline() -> Pipeline:
    return Pipeline(
        [
            node(
                func=prepare_data,
                inputs={
                    "input_file": "params:raw_dataset_path",
                    "output_file": "params:processed_dataset_path",
                },
                outputs=None,
                name="prepare_data_node",
            ),
        ]
    )


def create_model_training_pipeline():
    return Pipeline(
        [
            node(
                func=train_model,
                inputs={
                    "processed_data_path": "params:processed_dataset_path",
                    "evaluation_metrics_path": "params:evaluation_metrics_path",
                    "model_path": "params:model_path",
                },
                outputs=None,
                name="train_model_node",
            ),
        ]
    )


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = create_prepare_data_pipeline()
    model_training_pipeline = create_model_training_pipeline()

    pipelines = {
        "data_processing": data_processing_pipeline,
        "model_training": model_training_pipeline,
        "__default__": data_processing_pipeline + model_training_pipeline,
    }

    return pipelines