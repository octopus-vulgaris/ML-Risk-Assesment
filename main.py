from operator import index
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

dataframe = pd.read_csv("data_raw_no_ml.csv")

#print(dataframe.shape)
print(dataframe.head())

val_dataframe = dataframe.sample(frac=0.2, random_state=13)
train_dataframe = dataframe.drop(val_dataframe.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)

def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("RISK")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)

train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)


from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import StringLookup


def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = StringLookup if is_string else IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature


# Categorical features encoded as integers

cat_features_names = [
    "marital_status",
    "children",
    "education",
    "work_conditions",
    "smoking",
    "physical_activity",
    "sport",
    "alcochol",
    "alc_freq",
    "diet",
    "grind",
    "nigth_work",
    "social_load",
    "work_in_hurry",
    "high_demands",
    "career_advancement",
    "working_environment",
    "insecure_about_work",
    "ergonomic_workpace",
    "restroom"
]

# Numerical features

num_features_names = [
    "age",
    "weight",
    "heigth",
    "experience" ,
    "sleep_nigth",
    "alc_portion",
    "IBM",
    "WORK",
    "STRESS",
    "HADS",
    "AMS",
    "MIEF-5"
]

def inputs(list, dtype=None):
    all_inputs = []
    for x in list:
        all_inputs.append(keras.Input(shape=(1,), name=x, dtype=dtype))
    return all_inputs

cat_f_inputs = inputs(cat_features_names, "int64")
num_f_inputs = inputs(num_features_names)

all_inputs = cat_f_inputs + num_f_inputs


def encode_cat_features(inputs):
    features_encoded = []
    for x in inputs:
        features_encoded.append(encode_categorical_feature(x, x.name, train_ds, False))
    return features_encoded

def encode_num_features(inputs):
    features_encoded = []
    for x in inputs:
        features_encoded.append(encode_numerical_feature(x, x.name, train_ds))
    return features_encoded    

cat_features_encoded = encode_cat_features(cat_f_inputs)
num_features_encoded = encode_num_features(num_f_inputs)

all_features_pre = cat_features_encoded + num_features_encoded

all_features = layers.concatenate(all_features_pre)


x = layers.Dense(32, activation="relu")(all_features)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(all_inputs, output)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

model.fit(train_ds, epochs=50, validation_data=val_ds)