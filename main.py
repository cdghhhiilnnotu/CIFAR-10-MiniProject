import keras
import cv2
from keras import regularizers
import pandas as pd
import os
import numpy as np
import math

data_x = []
for index in range(len(os.listdir('Data/train'))):
    img = cv2.imread(f'Data/train/{index + 1}.png')/255.
    data_x.append(img)
data_x = np.array(data_x)
data_x = data_x.astype(np.float32)
x_train, x_test = data_x[:math.ceil(data_x.shape[0]*0.7)], data_x[math.ceil(data_x.shape[0]*0.7):]

labels = pd.read_csv('Data/trainLabels.csv')
data_y = pd.factorize(labels['label'])[0]
data_y = data_y.astype(np.float32)
y_train, y_test = data_y[:math.ceil(data_y.shape[0]*0.7)], data_y[math.ceil(data_y.shape[0]*0.7):]


inputs = keras.Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))

x = keras.layers.Conv2D(32,(3,3),activation='relu')(inputs)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
x = keras.layers.Dropout(0.25)(x)

x = keras.layers.Conv2D(64,(3,3),activation='relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
x = keras.layers.Dropout(0.25)(x)

x = keras.layers.Conv2D(128,(3,3),activation='relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
x = keras.layers.Dropout(0.25)(x)

x = keras.layers.Flatten()(x)
x = keras.layers.Dense(512,activation='relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.5)(x)

outputs = keras.layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=2)

model.evaluate(x_test, y_test, batch_size=32, verbose=2)

model.save('model-3.keras')





 