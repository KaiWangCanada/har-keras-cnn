# Compatibility layer between Python 2 and Python 3

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import confusion_matrix

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Reshape, GlobalAveragePooling1D
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.utils import np_utils
from keras import optimizers


def read_data(file_path):
    df = pd.read_csv(file_path, index_col=[0])

    df_stock = df[df['minor'] == 'VMC'][['minor', 'Date', '0', '1', '2', 'f1d_label']]
    return df_stock


def create_segments_and_labels(df, time_steps, step, label_name):
    # Number of steps to advance in each iteration
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['0'].values[i: i + time_steps]
        ys = df['1'].values[i: i + time_steps]
        zs = df['2'].values[i: i + time_steps]

        # use last days f1d_binary as label
        label = df[label_name].values[i + time_steps - 1]

        # # for testing
        # from numpy import random
        # label = random.randint(2)

        segments.append([xs, ys, zs])
        labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels


# ------- THE PROGRAM TO LOAD DATA AND TRAIN THE MODEL -------

'''params'''
# n features
N_FEATURES = 3

# The number of steps within one time segment
TIME_PERIODS = 60
# The steps to take from one segment to the next; if this value is equal to
# TIME_PERIODS, then there is no overlap between the segments
STEP_DISTANCE = 1

filter_height = 3
filter_depths = [2, 4, 8, 16]
drop_out = 0.5

# Hyper-parameters
BATCH_SIZE = 400
EPOCHS = 600

print("\n--- Load, inspect and transform data ---\n")

# Load data set containing all the data from csv
df = read_data('Data/temp_ret_3_30_stocks.csv')

# Define column name of the label vector
# LABEL = 'f1d_binary'
LABEL = 'f1d_label'

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

print("\n--- Create neural network model ---\n")

# 1D CNN neural network
model_m = Sequential()
model_m.add(Reshape((TIME_PERIODS, num_features), input_shape=(input_shape,)))  # 0
model_m.add(Conv1D(filter_depths[0], filter_height, activation='tanh', input_shape=(TIME_PERIODS, num_features)))  # 1
model_m.add(Conv1D(filter_depths[1], filter_height, activation='tanh'))  # 2
# model_m.add(MaxPooling1D(3))
model_m.add(AveragePooling1D(N_FEATURES))  # 3
model_m.add(Conv1D(filter_depths[2], filter_height, activation='tanh'))  # 4
model_m.add(Conv1D(filter_depths[3], filter_height, activation='tanh'))  # 5
model_m.add(GlobalAveragePooling1D())  # 6
model_m.add(Dropout(drop_out))  # 7
model_m.add(Dense(1))  # 8
print(model_m.summary())

model_m.compile(optimizer=optimizers.Adam(lr=1e-04), loss='mean_squared_error')

print("\n--- Fit the model ---\n")

# The EarlyStopping callback monitors training accuracy:
# if it fails to improve for two consecutive epochs,
# training stops early
callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='models_regression/best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=3)
]

# Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
history = model_m.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)

print('history:')
print(history.history)

plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], "b", label="Loss of training data")
plt.plot(history.history['val_loss'], "r", label="Loss of validation data")
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()

# print('\ncheck layers')
print(model_m.layers)

# print ('\ncheck layer before softmax output')
layer_models = []
for i in range(len(model_m.layers)):
    layer_models.append(Model(inputs=model_m.input, outputs=model_m.layers[i].output))

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

# y_test = np_utils.to_categorical(y_test, num_classes)

score = model_m.evaluate(x_test, y_test, verbose=1)

# print("\nAccuracy on test data: %0.2f" % score[1])
# print("\nLoss on test data: %0.2f" % score[0])
print('score:')
print(score)

y_pred_test = model_m.predict(x_test)
print('predict: ')
print(y_pred_test.tostring())

# print('\n layer before softmax output:')
# print(layer_models[7].predict(x_test))
