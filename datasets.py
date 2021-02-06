import tensorflow as tf
import numpy as np
import os
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage


class DataManager:
    def __init__(self):
        self._feature = FeatureManager()
        self._example = ExampleManager(self._feature)
        self._image = ImageHelper(self._example, self._feature)
        self._dataset = DatasetHelper(self._example, self._feature)
        self._text = TextHelper()
        self._csv = CSVHelper()
        self._utils = DataUtils()
        self._tfrecord = TFRecordManager()

    def create_feature_contents(self, val_map):
        feature_contents = []
        for key, vals in val_map.items():
            for val in vals:
                feature_contents.append(FeatureContent(key, val))

        return feature_contents

    def create_feature_defs(self, val_map):
        feature_defs = []
        for key, val in val_map.items():
            feature_defs.append(FeatureDef(key, val))

        return feature_defs


class FeatureManager:
    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()

        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def serialize_to_string(self, feature):
        return feature.SerializeToString()


class ImageHelper:
    def __init__(self, example_manager=None, feature_manager=None):
        if example_manager is None:
            self._example_manager = ExampleManager()
        else:
            self._example_manager = example_manager
        if feature_manager is None:
            self._feature_manager = FeatureManager()
        else:
            self._feature_manager = feature_manager

    def get_image_data_generator(self, **kwargs):
        return tf.keras.preprocessing.image.ImageDataGenerator(**kwargs)

    def get_images_and_labels_gen(image_data_generator, directory):
        images, labels = next(image_data_generator.flow_from_directory(directory))
        return images, labels

    def get_image_string(self, filename):
        return open(filename, 'rb').read()

    def write_images(self, record_filename, image_labels):
        with tf.io.TFRecordWriter(record_filename) as writer:
            for filename, label in image_labels.items():
                image_string = open(filename, 'rb').read()
                tf_example = self._example_manager.image_example(image_string, label)
                writer.write(tf_example.SerializeToString())

    def parse_image_function(self, example_proto, feature_description):
        return tf.io.parse_single_example(example_proto, feature_description)

    def parse_image(self, filename, dims=[128, 128]):
        parts = tf.strings.split(filename, os.sep)
        label = parts[-2]
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, dims)
        return image, label

    def random_rotate_image(self, image):
        image = ndimage.rotate(image, np.random.uniform(-30, 30), reshape=False)
        return image

    def show_image(self, image, label):
        plt.figure()
        plt.imshow(image)
        plt.title(label.numpy().decode('utf-8'))
        plt.axis('off')
        plt.show()

    def read_images(self, dataset):
        image_feature_description = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
        }

        return dataset.map(parse_image_function)

    def image_example(self, image_string, label):
        image_shape = tf.image.decode_jpeg(image_string).shape
        feature = {
            'height': self._feature_manager._int64_feature(image_shape[0]),
            'width': self._feature_manager._int64_feature(image_shape[1]),
            'depth': self._feature_manager._int64_feature(image_shape[2]),
            'label': self._feature_manager._int64_feature(label),
            'image_raw': self._feature_manager._bytes_feature(image_string),
        }

        return tf.train.Example(features=tf.train.Features(feature=feature))


class DatasetHelper:
    def __init__(self, example_manager=None, feature_manager=None):
        if example_manager is None:
            self._example_manager = ExampleManager()
        else:
            self._example_manager = example_manager
        if feature_manager is None:
            self._feature_manager = FeatureManager()
        else:
            self._feature_manager = feature_manager

    def create_dataset(self, values):
        return tf.data.Dataset.from_tensor_slices(values)

    def serialize_dataset(self, dataset, generator=False):
        if generator:
            gen = self.get_serialization_generator()
            return tf.data.Dataset.from_generator(
                gen, output_types=tf.string, output_shapes=()
            )
        return dataset.map(self._example_manager.tf_serialize_example)

    def get_serialization_generator(self, dataset):
        def generator():
            for features in dataset:
                yield self._example_manager.serialize_example(*features)

        return generator

    def create_dataset_from_generator(self, gen, args=None):
        if args is None:
            return tf.data.Dataset.from_generator(gen)
        return tf.data.Dataset.from_generator(gen, args=args)

    def parse_batch(self, dataset):
        return dataset.map(parse_function)

    def get_balanced_dataset(self, datasets, weights, batch_size):
        return tf.data.experimental.sample_from_datasets(
            datasets,
            weights
        ).batch(batch_size)


class DataUtils:
    def get_file(self, fname, origin, **kwargs):
        return tf.keras.utils.get_file(fname, origin, **kwargs)

    def get_file_data_and_label(self, file_path):
        label = tf.strings.split(file_path, os.sep)[-2]
        return tf.io.read_file(file_path), label

    def get_counter(self):
        return tf.data.experimental.Counter()

    def get_distribution(self, counts, batch):
        features, labels = batch
        class_1 = labels == 1
        class_1 = tf.cast(class_1, tf.int32)

        class_0 = labels == 0
        class_0 = tf.cast(class_0, tf.int32)

        counts['class_0'] += tf.reduce_sum(class_0)
        counts['class_1'] += tf.reduce_sum(class_1)

        return counts


class TextHelper:
    def get_text_line_dataset(self, file_paths):
        return tf.data.TextLineDataset(file_paths)


class CSVHelper:
    def make_csv_dataset(self, file_pattern, batch_size, **kwargs):
        return tf.data.experimental.make_csv_dataset(file_pattern, batch_size, **kwargs)

    def get_dataframe(self, file_path, **kwargs):
        return pd.read_csv(file_path, **kwargs)


class Dataset:
    def take(self):
        pass

    def shuffle(self):
        pass

    def padded_batch(self):
        pass

    def interleave(self):
        pass

    def filter(self):
        pass

    def map(self):
        pass

    def range(self):
        pass

    def zip(self):
        pass

    def batch(self):
        pass

    def repeat(self):
        pass

    def window(self):
        pass

    def unbatch(self):
        pass


class FeatureContent:
    def __init__(self, name, content):
        self.name = name
        self.content = content


class FeatureDef:
    def __init__(self, name, type):
        self.name = name
        self.type = type


class ExampleManager:
    def __init__(self, feature_manager=None):
        if feature_manager is None:
            self._feature_manager = FeatureManager()
        else:
            self._feature_manager = feature_manager
        self.configuration = {}

        self.FEATURE_TYPE_MAPPING = {
            'int64': self._feature_manager._int64_feature,
            'bytes': self._feature_manager._bytes_feature,
            'float': self._feature_manager._float_feature
        }

        self.FEATURE_DESC_MAPPING = {
            'int64': tf.io.FixedLenFeature([], tf.int64, default_value=0),
            'bytes': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'float': tf.io.FixedLenFeature([], tf.float32, default_value=0.0)
        }

    def configure(self, feature_defs):
        for feature_def in feature_defs:
            self.configuration[feature_def.name] = feature_def.type

    def serialize_example(self, features):
        """
        Creates a tf.train.Example message ready to be written to a file.
        """
        feature_map = self.create_feature_map(features)
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature_map))

        return self._feature_manager.serialize_to_string(example_proto)

    def read_batch_examples(self, filenames):
        dataset = self.read_records(filenames)
        examples = []
        for record in dataset.take(1):
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            examples.append(example)
        return examples

    def write_batch_examples(self, filename, features):
        with tf.io.TFRecordWriter(filename) as writer:
            for i in range(len(features_set)):
                example = self.serialize_example(
                    features_set[i]
                )
                writer.write(example)


    def deserialize_example(self, serialized_example):
        return tf.train.Example.FromString(serialized_example)

    def get_feature_type(self, name):
        return self.configuration[name]

    def get_feature_type_function(self, name):
        feature_type = self.get_feature_type(name)
        return FEATURE_TYPE_MAPPING[feauture_type]

    def create_feature_map(self, features):
        feature_map = {
            feature.name : self.get_feature_type_function(feature.name)(feature.content)
            for feature in features
        }

        return feature_map

    def tf_serialize_example(self, features):
        tf_string = tf.py_function(
            self.serialize_example(
                serialize_example,
                tuple(features),
                tf.string
            )
        )

        return tf.reshape(tf_string, ())

    def get_feature_desc_function(self, type):
        return self.FEATURE_DESC_MAPPING[type]

    def parse_function(self, example_proto, feature_defs):
        feature_description = {
            feature_def.name : self.get_feature_desc_function(feature.type)
            for feature_def in feature_defs
        }
        return tf.io.parse_single_example(example_proto, feature_description)


class TFRecordManager:
    def write_record(self, filename, serialized_dataset):
        writer = tf.data.experimental.TFRecordWriter(filename)
        writer.write(serialized_dataset)

    def read_records(self, filenames):
        return tf.data.TFRecordDataset(filenames)
