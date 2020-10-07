import os
import numpy as np
import tensorflow as tf
import sklearn.metrics as sm
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random

random.seed(1618)
np.random.seed(1618)
# tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ALGORITHM = "guesser"
# ALGORITHM = "tf_net"
ALGORITHM = "tf_conv"

# DATASET = "mnist_d"
# DATASET = "mnist_f"
DATASET = "cifar_10"
# DATASET = "cifar_100_f"
# DATASET = "cifar_100_c"

if DATASET == "mnist_d":
	NUM_CLASSES = 10
	IMAGE_HEIGHT = 28
	IMAGE_WIDTH = 28
	IMAGE_Z = 1
	IMAGE_SIZE = 784

elif DATASET == "mnist_f":
	NUM_CLASSES = 10
	IMAGE_HEIGHT = 28
	IMAGE_WIDTH = 28
	IMAGE_Z = 1
	IMAGE_SIZE = 784

elif DATASET == "cifar_10":
	NUM_CLASSES = 10
	IMAGE_HEIGHT = 32
	IMAGE_WIDTH = 32
	IMAGE_Z = 3
	IMAGE_SIZE = 3072

elif DATASET == "cifar_100_f":
	pass  # TODO: Add this case.

elif DATASET == "cifar_100_c":
	pass  # TODO: Add this case.


# =========================<Classifier Functions>================================

def guesserClassifier(xTest):
	ans = []
	for entry in xTest:
		pred = [0] * NUM_CLASSES
		pred[random.randint(0, 9)] = 1
		ans.append(pred)
	return np.array(ans)


def buildTFNeuralNet(x, y, eps=6):
	model = tf.keras.models.Sequential(
		[tf.keras.layers.Flatten(), tf.keras.layers.Dense(256, activation=tf.nn.relu),
		tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	model.fit(x, y, epochs=eps)
	return model


def buildTFConvNet(x, y, eps=10, dropout=True, dropRate=0.2):

	if DATASET == "mnist_d" or DATASET == "mnist_f":
		model = keras.Sequential()
		inShape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_Z)
		model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=inShape))
		model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation=tf.nn.relu))
		model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
		model.add(keras.layers.Flatten())
		model.add(keras.layers.Dense(256, activation=tf.nn.relu))
		model.add(keras.layers.Dense(NUM_CLASSES, activation="softmax"))
		model.add(keras.layers.Dropout(0.25))

		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		model.fit(x, y, epochs=7)

		return model

	elif DATASET == "cifar_10":
		model = keras.Sequential()
		inShape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_Z)
		model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=inShape))
		model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation=tf.nn.relu))
		model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
		model.add(keras.layers.Dropout(0.5))
		model.add(keras.layers.Flatten())
		model.add(keras.layers.BatchNormalization())
		model.add(keras.layers.Dense(512, activation=tf.nn.relu))
		model.add(keras.layers.Dense(NUM_CLASSES, activation="softmax"))

		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		model.fit(x, y, epochs=7, batch_size=100)

		return model

# =========================<Pipeline Functions>==================================

def getRawData():
	if DATASET == "mnist_d":
		mnist = tf.keras.datasets.mnist
		(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
	elif DATASET == "mnist_f":
		mnist = tf.keras.datasets.fashion_mnist
		(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
	elif DATASET == "cifar_10":
		cifar10 = tf.keras.datasets.cifar10
		(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
	elif DATASET == "cifar_100_f":
		pass  # TODO: Add this case.
	elif DATASET == "cifar_100_c":
		pass  # TODO: Add this case.
	else:
		raise ValueError("Dataset not recognized.")
	print("Dataset: %s" % DATASET)
	print("Shape of xTrain dataset: %s." % str(xTrain.shape))
	print("Shape of yTrain dataset: %s." % str(yTrain.shape))
	print("Shape of xTest dataset: %s." % str(xTest.shape))
	print("Shape of yTest dataset: %s." % str(yTest.shape))
	return ((xTrain, yTrain), (xTest, yTest))


def preprocessData(raw):
	((xTrain, yTrain), (xTest, yTest)) = raw

	# Range reduction here (0-255 ==> 0.0-1.0).
	xTrain, xTest = xTrain / 255.0, xTest / 255.0

	if ALGORITHM != "tf_conv":
		xTrainP = xTrain.reshape((xTrain.shape[0], IMAGE_SIZE))
		xTestP = xTest.reshape((xTest.shape[0], IMAGE_SIZE))
	else:
		xTrainP = xTrain.reshape((xTrain.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_Z))
		xTestP = xTest.reshape((xTest.shape[0], IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_Z))

	yTrainP = to_categorical(yTrain, NUM_CLASSES)
	yTestP = to_categorical(yTest, NUM_CLASSES)
	print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
	print("New shape of xTest dataset: %s." % str(xTestP.shape))
	print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
	print("New shape of yTest dataset: %s." % str(yTestP.shape))
	return ((xTrainP, yTrainP), (xTestP, yTestP))


def trainModel(data):
	xTrain, yTrain = data

	if ALGORITHM == "guesser":
		return None  # Guesser has no model, as it is just guessing.

	elif ALGORITHM == "tf_net":
		print("Building and training TF_NN.")
		return buildTFNeuralNet(xTrain, yTrain)

	elif ALGORITHM == "tf_conv":
		print("Building and training TF_CNN.")
		return buildTFConvNet(xTrain, yTrain)

	else:
		raise ValueError("Algorithm not recognized.")


def runModel(data, model):
	if ALGORITHM == "guesser":
		return guesserClassifier(data)

	elif ALGORITHM == "tf_net":
		print("Testing TF_NN.")

		# get predictions
		preds = model.predict(data)

		# convert to only 0s and 1s
		for i in range(preds.shape[0]):
			oneHot = [0] * NUM_CLASSES
			oneHot[np.argmax(preds[i])] = 1
			preds[i] = oneHot
		return preds

	elif ALGORITHM == "tf_conv":
		print("Testing TF_CNN.")
		preds = model.predict(data)
		for i in range(preds.shape[0]):
			oneHot = [0] * NUM_CLASSES
			oneHot[np.argmax(preds[i])] = 1
			preds[i] = oneHot
		return preds

	else:
		raise ValueError("Algorithm not recognized.")


def evalResults(data, preds):
	xTest, yTest = data
	acc = 0

	for i in range(preds.shape[0]):
		if np.array_equal(preds[i], yTest[i]):
			acc = acc + 1

	accuracy = float(acc) / preds.shape[0]
	print("Classifier algorithm: %s" % ALGORITHM)
	print("Classifier accuracy: %f%%" % (accuracy * 100))

	# Getting predicted and actual vals for confusion matrix and f1 score

	pVals = []
	acVals = []

	for i in range(preds.shape[0]):
		pVals.append(np.argmax(preds[i]))
		acVals.append(np.argmax(yTest[i]))

	print("Confusion Matrix")
	print(sm.confusion_matrix(acVals, pVals))

	print("F1 Score")
	print(sm.f1_score(acVals, pVals, average='micro'))


# =========================<Main>================================================

def main():
	raw = getRawData()
	data = preprocessData(raw)
	model = trainModel(data[0])
	preds = runModel(data[1][0], model)
	evalResults(data[1], preds)


if __name__ == '__main__':
	main()
