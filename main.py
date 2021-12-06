import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

dataframe = pd.read_csv("data_raw.csv")

#print(dataframe.shape)
print(dataframe.head())

val_dataframe = dataframe.sample(frac=0.2, random_state=13)
train_dataframe = dataframe.drop(val_dataframe.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)

