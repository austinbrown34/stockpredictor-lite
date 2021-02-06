import os
import time
from tensorflow.keras.layers import LSTM
from transformers import StockPredictorTransformer


LOSS = 'mse'
UNITS = 256
CELL = LSTM
N_LAYERS = 3
DROPOUT = 0.4
OPTIMIZER = "rmsprop"
BATCH_SIZE = 64
EPOCHS = 100
MODEL_NAME = "stockpredictor"


FILE_PATTERN = "csv_data/*.csv"
SELECT_COLUMNS = ["volume", "open", "high", "low", "close"]
LABEL_NAME = "close"
TRANSFORM_PARAMS = {
    "label": LABEL_NAME,
    "num_examples": 100,
    "n_steps_range": [1, 100],
    "lookup_step_range": [1, 100]
}
TRANSFORMER = StockPredictorTransformer
