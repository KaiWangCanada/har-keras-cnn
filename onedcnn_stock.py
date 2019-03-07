# Compatibility layer between Python 2 and Python 3

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
from random import random

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils


def read_data(file_path):
    df = pd.read_csv(file_path, index_col=[0])
    df['f1d_binary'] = np.where(df['f1d_label'] > 0, 1, 0)

    df_stock = df[df['minor'] == 'VMC'][['minor', 'Date', '0', '1', '2', 'f1d_binary']]
    return df_stock


def create_segments_and_labels(df, time_steps, step, label_name):

    # n features
    N_FEATURES = 3
    # Number of steps to advance in each iteration
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['0'].values[i: i + time_steps]
        ys = df['1'].values[i: i + time_steps]
        zs = df['2'].values[i: i + time_steps]

        # use last days f1d_binary as label
        label = df[label_name].values[i + time_steps - 1]
        segments.append([xs, ys, zs])
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels

# ------- THE PROGRAM TO LOAD DATA AND TRAIN THE MODEL -------

# The number of steps within one time segment
TIME_PERIODS = 60
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
STEP_DISTANCE = 1

print("\n--- Load, inspect and transform data ---\n")

# Load data set containing all the data from csv
df = read_data('Data/temp_ret_3_30_stocks.csv')

# Define column name of the label vector
LABEL = 'f1d_binary'

print("\n--- Reshape the data into segments ---\n")

# Differentiate between test set and training set
df_test = df[df['Date'] > '2019-02-24T09:30:00Z']
df_train = df[df['Date'] <= '2019-02-24T09:30:00Z']

# Reshape the training data into segments
# so that they can be processed by the network
x_train, y_train = create_segments_and_labels(df_train,
                                              TIME_PERIODS,
                                              STEP_DISTANCE,
                                              LABEL)

print("\n--- Reshape data to be accepted by Keras ---\n")

# Inspect x data
print('x_train shape: ', x_train.shape)
# Displays (134, 80, 3)
print(x_train.shape[0], 'training samples')
# Displays 134 train samples

# Inspect y data
print('y_train shape: ', y_train.shape)
# Displays (134,)

# Set input & output dimensions
num_time_periods, num_features = x_train.shape[1], x_train.shape[2]
num_classes = 2

# Set input_shape / reshape for Keras
# todo: remove
# Remark: acceleration data is concatenated in one array in order to feed
# it properly into coreml later, the preferred matrix of shape [80,3]
# cannot be read in with the current version of coreml (see also reshape
# layer as the first layer in the keras model)
input_shape = (num_time_periods * num_features)
x_train = x_train.reshape(x_train.shape[0], input_shape)

print('x_train shape:', x_train.shape)
# x_train shape: (20869, 240)
print('input_shape:', input_shape)
# input_shape: (240)

# Convert type for Keras otherwise Keras cannot process the data
x_train = x_train.astype("float32")
y_train = y_train.astype("float32")

# %%

# One-hot encoding of y_train labels (only execute once!)
y_train = np_utils.to_categorical(y_train, num_classes)
print('New y_train shape: ', y_train.shape)
# (20868, 6)

# %%

print("\n--- Create neural network model ---\n")

# 1D CNN neural network
model_m = Sequential()
model_m.add(Reshape((TIME_PERIODS, num_features), input_shape=(input_shape,)))
model_m.add(Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_features)))
model_m.add(Conv1D(100, 10, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(Conv1D(160, 10, activation='relu', data_format='channels_first'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(num_classes, activation='softmax'))
print(model_m.summary())

print("\n--- Fit the model ---\n")

# The EarlyStopping callback monitors training accuracy:
# if it fails to improve for two consecutive epochs,
# training stops early
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
]

model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])

# Hyper-parameters
BATCH_SIZE = 400
EPOCHS = 60

# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
history = model_m.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)

print("\n--- Check against test data ---\n")

x_test, y_test = create_segments_and_labels(df_test,
                                            TIME_PERIODS,
                                            STEP_DISTANCE,
                                            LABEL)

# Set input_shape / reshape for Keras
# todo: remove
x_test = x_test.reshape(x_test.shape[0], input_shape)

x_test = x_test.astype("float32")
y_test = y_test.astype("float32")

y_test = np_utils.to_categorical(y_test, num_classes)

score = model_m.evaluate(x_test, y_test, verbose=1)

print("\nAccuracy on test data: %0.2f" % score[1])
print("\nLoss on test data: %0.2f" % score[0])

y_pred_test = model_m.predict(x_test)
# Take the class with the highest probability from the test predictions
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test, axis=1)

print("\n--- Classification report for test data ---\n")
print(classification_report(max_y_test, max_y_pred_test))