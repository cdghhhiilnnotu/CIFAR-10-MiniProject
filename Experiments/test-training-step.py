import keras
import cv2
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

model = keras.models.load_model('model-1.keras')

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=2)

model.evaluate(x_test, y_test, batch_size=128, verbose=2)

model.save('model-1.keras')
