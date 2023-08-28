import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

data_x = []
for index in range(len(os.listdir('Data/test'))):
    img = cv2.imread(f'Data/test/{index + 1}.png')/255.
    data_x.append(img)
data_x = np.array(data_x)
data_x = data_x.astype(np.float32)

model = keras.models.load_model('model-2.keras')

predictions = model.predict(data_x)
labels = ['frog', 'truck', 'deer', 'automobile', 'bird', 'horse', 'ship', 'cat', 'dog', 'airplane']
indexImg = 109
plt.imshow(data_x[index])
plt.xlabel(f'Predictions: {labels[np.argmax(predictions[index])]}')
plt.show()


