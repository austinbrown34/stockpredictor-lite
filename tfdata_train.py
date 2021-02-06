import tensorflow as tf
from datasets import DataManager
import glob
import pandas as pd
import numpy as np
from models import get_model
from params import *
import os
from sklearn.externals import joblib
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


class DatasetBuilder:
    def __init__(self):
        self.manager = DataManager()
        self.extra = {}

    def get_default_config(self):
        file_pattern = FILE_PATTERN
        select_columns = SELECT_COLUMNS
        label_name = LABEL_NAME
        transformer = StockPredictorTransformer
        transformer_params = TRANSFORM_PARAMS
        return [
            file_pattern,
            select_columns,
            label_name,
            transformer,
            transformer_params
        ]

    def get_dataframes(self, csv_paths, select_columns):
        return [
            self.manager._csv.get_dataframe(
                csv_path,
                index_col=0
            )
            for csv_path in csv_paths
        ]

    def get_extra(self):
        return self.extra

    def datasets_from_csvs(
        self,
        file_pattern,
        select_columns,
        label_name,
        transformer,
        transformer_params,
        **kwargs
    ):
        csv_paths = glob.glob(file_pattern)
        dataframes = self.get_dataframes(csv_paths, select_columns)
        frame = pd.concat(dataframes, axis=0, ignore_index=False)
        input_train, input_test, target_train, target_test, extra = transformer(
            frame, **transformer_params
        ).transform()
        max_seq = input_train.bounding_shape()[-1].numpy()
        extra.update({
            'max_seq': max_seq
        })
        self.extra.update(extra)
        data = [
            (input_train, target_train),
            (input_test, target_test)
        ]
        return [
            self.manager._dataset.create_dataset((d[0], d[1]))
            for d in data
        ]


if __name__ == '__main__':
    dataset_builder = DatasetBuilder()
    config = dataset_builder.get_default_config()
    datasets = dataset_builder.datasets_from_csvs(*config)
    extra = dataset_builder.get_extra()
    max_seq = extra['max_seq']
    column_scaler = extra['column_scaler']
    model = get_model(
        max_seq,
        UNITS,
        CELL,
        N_LAYERS,
        DROPOUT,
        LOSS,
        OPTIMIZER
    )
    model_path = os.path.join("results", MODEL_NAME) + ".h5"

    if os.path.exists(model_path):
        model.load_weights(model_path)

    train_dataset = datasets[0].batch(BATCH_SIZE)
    test_dataset = datasets[1].batch(BATCH_SIZE)

    checkpointer = ModelCheckpoint(os.path.join("results", MODEL_NAME), save_weights_only=True, save_best_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir=os.path.join("logs", MODEL_NAME))

    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=test_dataset,
        callbacks=[checkpointer, tensorboard],
        verbose=1
    )

    model.save(os.path.join("results", MODEL_NAME) + ".h5")

    joblib.dump(column_scaler, os.path.join("results", MODEL_NAME) + "_column_scaler.pkl")
    np.save(os.path.join("results", MODEL_NAME) + "_max_seq.npy", max_seq)
