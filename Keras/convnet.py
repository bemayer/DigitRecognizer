import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras import layers
from keras import models

data_train = pd.read_csv("./Data/data_train.csv").filter(regex=("pix*")).to_numpy().reshape(1500, 15, 16, 1)
data_label = np_utils.to_categorical(pd.read_csv("./Data/data_train.csv").filter(regex=("class")).to_numpy(), 10)

model = models.Sequential()
model.add(layers.Conv2D(16, (3,3), activation='relu', input_shape=(15, 16, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(data_train, data_label, epochs=20, batch_size=64)
