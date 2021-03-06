import numpy as np
from utils import mnist_reader
from utils import plot_utils

from keras.utils import np_utils
from models.cnnmodel import CNNModel
from models.alexnetmodel import AlexNetModel
from models.vgg16model import VGG16Model
# from models.fcmodel_solution import FCModel

from argparse import ArgumentParser


# loading the data set and convert to correct format and scale
X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

# some more details: https://github.com/zalandoresearch/fashion-mnist

# image size
img_rows, img_cols = 28, 28

nb_classes = 10
# 0	T-shirt/top
# 1	Trouser
# 2	Pullover
# 3	Dress
# 4	Coat
# 5	Sandal
# 6	Shirt
# 7	Sneaker
# 8	Bag
# 9	Ankle boot

# uncomment for debugging
# show 9 grayscale images as examples of the data set
# ------- start show images ----------
# import sys
# from matplotlib import pyplot as plt
# for i in range(9):
#     plt.subplot(3,3,i+1)
#     plt.imshow(X_train[i].reshape(28,28), cmap='gray', interpolation='none')
#     plt.title("Class {}".format(y_train[i]))
#
# plt.show()
# sys.exit()
# ------- end show images ----------




# converts a class vector (list of labels in one vector (as for SVM)
# to binary class matrix (one-n-encoding)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# we need to reshape the input data to fit keras.io input matrix format
# X_train, X_test = FCModel.reshape_input_data(X_train, X_test)
X_train, X_test = CNNModel.reshape_input_data(X_train, X_test)

# which model
parser = ArgumentParser()
parser.add_argument("-m", "--model_name", help="name ['default','alex','vgg16'] default:{'default'} ")
parser.add_argument("-e", "--epochs", help="number of epochs default:{10}", type=int)
args = parser.parse_args()
if args.model_name:
    suffix = args.model_name
else:
    suffix = "default"
# suffix = "alex"

# hyperparameter
if args.epochs:
    nb_epoch = args.epochs
else:
    nb_epoch = 10
batch_size = 128


model = None
if suffix == "default":
    model = CNNModel.load_model(nb_classes)
if suffix == "alex":
    model = AlexNetModel.load_model(nb_classes)
if suffix == "vgg16":
    model = VGG16Model.load_model(nb_classes)

# model = FCModel.load_model(nb_classes)
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))


score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

plot_utils.plot_model_history(history,suffix)
plot_utils.plot_result_examples(model, X_test, y_test, img_rows, img_cols,suffix)

