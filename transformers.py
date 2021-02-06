import random
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


class StockPredictorTransformer:
    def __init__(self, df, **kwargs):
        self.df = df
        self.label = kwargs.get("label")
        self.num_examples = kwargs.get("num_examples")
        self.n_steps_range = kwargs.get("n_steps_range")
        self.lookup_step_range = kwargs.get("lookup_step_range")

    def _get_example(self, n_steps, lookup_step):
        max_index = (self.df.shape[0] - 1) - (lookup_step + n_steps)
        start = random.randint(0, max_index)
        return {
            'n_steps': n_steps,
            'start': start,
            'past': self.df.iloc[start:start + n_steps],
            'lookup_step': lookup_step,
            'future': self.df.iloc[start + lookup_step + n_steps - 1:start + lookup_step + n_steps]
        }

    def _get_examples(self, num_examples, n_steps_range, lookup_step_range):
        examples = []
        for _ in range(num_examples):
            try:
                n_steps = random.randint(*n_steps_range)
                lookup_step = random.randint(*lookup_step_range)
                example = self._get_example(n_steps, lookup_step)
                past_with_lookup = example['past'].copy()
                past_with_lookup['lookup_step'] = lookup_step
                past_with_lookup_values = past_with_lookup.values.tolist()
                examples.append(
                    [
                        past_with_lookup_values,
                        example['future'][self.label]
                    ]
                )
            except Exception as e:
                print(str(e))
                pass

        return examples

    def _separate_inputs_and_targets(self, examples):
        inputs, targets = [], []
        for input, target in examples:
            inputs.append(np.array(input))
            targets.append(np.array(target))

        return inputs, targets


    def _normalize_df(self):
        column_scaler = {}
        for column in self.df.columns.tolist():
            scaler = preprocessing.MinMaxScaler()
            self.df[column] = scaler.fit_transform(np.expand_dims(self.df[column].values, axis=1))
            column_scaler[column] = scaler
        self.column_scaler = column_scaler


    def get_column_scaler(self):
        return self.column_scaler

    def transform(self):
        self._normalize_df()
        extra = {
            "column_scaler": self.get_column_scaler()
        }
        examples = self._get_examples(
            self.num_examples,
            self.n_steps_range,
            self.lookup_step_range
        )
        inputs, targets = self._separate_inputs_and_targets(examples)
        input_train, input_test, target_train, target_test = train_test_split(
            inputs,
            targets,
            test_size=0.2,
            shuffle=True
        )
        input_train = tf.ragged.constant(input_train)
        input_test = tf.ragged.constant(input_test)
        return input_train, input_test, target_train, target_test, extra
