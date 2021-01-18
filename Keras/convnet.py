# ------------------------------------------------------------------------------
# Libraries used
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
from keras.utils import np_utils
from keras import layers
from keras import models
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Training function with logs to csv files
def train(name, model, data_train, data_test, n_epochs):
	data_train_label = (np_utils.to_categorical(
		pd.read_csv("./Data/data_train.csv")
		.filter(regex=("class")).to_numpy(), 10))
	data_test_label = np_utils.to_categorical(
		pd.read_csv("./Data/data_test_completed.csv")
		.filter(regex=("class")).to_numpy(), 10)
	model.compile(optimizer='adam', loss='categorical_crossentropy',
		metrics=['accuracy'])
	history = model.fit(data_train, data_train_label, epochs=n_epochs,
	batch_size=64, validation_data=(data_test, data_test_label))
	epo = np.array(history.epoch)
	acc_train = np.array(history.history["accuracy"])
	acc_test = np.array(history.history["val_accuracy"])
	log = np.c_[epo, acc_train, acc_test]
	tim = time.strftime("_%Y%m%d-%H%M%S")
	np.savetxt("./Log/" + name + tim + ".csv", log, delimiter=",")
	pred = model.predict(data_test)
	np.savetxt("./Log/pred_" + name + tim + ".csv", pred, delimiter=",")
	test_loss, test_acc = model.evaluate(data_test, data_test_label)
	return(test_acc)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Model 0, 98.40%
def model_0():
	data_train = (pd.read_csv("./Data/data_train.csv")
		.filter(regex=("pix*")).to_numpy() / 6)
	model = models.Sequential()
	model.add(layers.Dense(120, activation="relu"))
	model.add(layers.Dense(10, activation="softmax"))
	data_test = (pd.read_csv("./Data/data_test_completed.csv")
		.filter(regex=("pix*")).to_numpy() / 6)
	acc_0 = train("M0", model, data_train, data_test, 50)
	print("Model 0:", acc_0)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Model 1, 98.40%
def model_1():
	data_train = (pd.read_csv("./Data/data_train.csv")
		.filter(regex=("pix*")).to_numpy() / 6)
	model = models.Sequential()
	model.add(layers.Dense(100, activation="relu"))
	model.add(layers.Dense(100, activation="relu"))
	model.add(layers.Dense(100, activation="relu"))
	model.add(layers.Dense(100, activation="relu"))
	model.add(layers.Dense(100, activation="relu"))
	model.add(layers.Dense(10, activation="softmax"))
	data_test = (pd.read_csv("./Data/data_test_completed.csv")
		.filter(regex=("pix*")).to_numpy() / 6)
	acc_1 = train("M1", model, data_train, data_test, 10)
	print("Model 1:", acc_1)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Model 2, 97.60%
def model_2():
	data_train = (pd.read_csv("./Data/data_train.csv")
		.filter(regex=("pix*")).to_numpy().reshape(1500, 15, 16, 1) / 6)
	model = models.Sequential()
	model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu",
		input_shape=(15, 16, 1)))
	model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
	model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
	model.add(layers.Flatten())
	model.add(layers.Dense(64, activation="relu"))
	model.add(layers.Dense(10, activation="softmax"))
	data_test = (pd.read_csv("./Data/data_test_completed.csv")
	.filter(regex=("pix*")).to_numpy().reshape(500, 15, 16, 1) / 6)
	acc_2 = train("M2", model, data_train, data_test, 10)
	model.summary()
	print("Model 2:", acc_2)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Model 3, 98.40%
def model_3(train_file):
	data_train = (pd.read_csv("./Data/data_train.csv")
		.filter(regex=("_")).to_numpy())
	model = models.Sequential()
	model.add(layers.Dense(643, activation="relu"))
	model.add(layers.Dense(200, activation="relu"))
	model.add(layers.Dense(200, activation="relu"))
	model.add(layers.Dense(200, activation="relu"))
	model.add(layers.Dense(200, activation="relu"))
	model.add(layers.Dense(10, activation="softmax"))
	data_test = (pd.read_csv("./Data/data_test_completed.csv")
		.filter(regex=("_")).to_numpy())
	acc_3 = train("M3", model, data_train, data_test, 100)
	print("Model 3:", acc_3)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Repeat 30 times the training
for i in range(30):
	model_0()
	model_1()
	model_2()
	model_3()
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Data Augmentation

# Elastic transform function
def elastic_transform(image, alpha_range, sigma, random_state=None):
	if random_state is None:
		random_state = np.random.RandomState(None)
	if np.isscalar(alpha_range):
		alpha = alpha_range
	else:
		alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])
	shape = image.shape
	dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
	dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
	x, y, z = (np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
		np.arange(shape[2]), indexing='ij'))
	indices = (np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1)),
		np.reshape(z, (-1, 1)))
	return map_coordinates(image, indices, order=1,
		mode='reflect').reshape(shape)

# ImageData Generator parameters
datagen = ImageDataGenerator(
		rotation_range = 10,
		width_shift_range=0.1,
		height_shift_range=0.1,
		shear_range=0,
		zoom_range=0.2,
		preprocessing_function=lambda x: elastic_transform(x, alpha_range=8
			, sigma=3)
	)

# Input data
data_train = (pd.read_csv("./Data/data_train.csv").filter(regex=("pix*"))
		.to_numpy().reshape(1500, 15, 16, 1) / 6)
data_train_label = (np_utils.to_categorical(
	pd.read_csv("./Data/data_train.csv").filter(regex=("class"))
	.to_numpy(), 10))
X_train = data_train
y_train = data_train_label
X_test = (pd.read_csv("./Data/data_test_completed.csv")
	.filter(regex=("pix*")).to_numpy() / 6)
y_test = np_utils.to_categorical(
	pd.read_csv("./Data/data_test_completed.csv")
	.filter(regex=("class")).to_numpy(), 10)

# Generate 15000 new images
for iter in range(10):
	augmented = datagen.flow(data_train, data_train_label, batch_size=1500,
		shuffle=False).next()
	for line in range(augmented[0].shape[0]):
		pix = augmented[0][line].reshape(240)
		image_RGB = np.array([[255-255*pix[i+15*j] for i in range(15)]
							for j in range(16)], dtype=np.uint8)
		image = Image.fromarray(image_RGB)
		image.save('./Data/Augmented/img_' + str(line) + "_" +
			str(iter) + '.jpg', 'JPEG')
	X_train = np.concatenate([X_train, augmented[0]], axis=0)
	y_train = np.concatenate([y_train, augmented[1]], axis=0)

# Model 1, 98.40%
model = models.Sequential()
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))
model.compile(optimizer='adam', loss='categorical_crossentropy',
	metrics=['accuracy'])
history = model.fit(X_train.reshape(X_train.shape[0], 240), y_train, epochs=100,
		batch_size=64, validation_data=(X_test, y_test))


# Model 2, 98.40%
model = models.Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu",
	input_shape=(15, 16, 1)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))
model.compile(optimizer='adam', loss='categorical_crossentropy',
	metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100,
		batch_size=64, validation_data=(X_test.reshape(500, 15, 16, 1), y_test))
# ------------------------------------------------------------------------------
