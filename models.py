import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Embedding


def get_model(max_seq, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop"):
    model = Sequential()
    model.add(Input(shape=[None, max_seq], dtype=tf.float32, ragged=True))
    for i in range(n_layers):
        if i == n_layers - 1:
            model.add(cell(units, return_sequences=False))
        else:
            model.add(cell(units, return_sequences=True))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    return model
