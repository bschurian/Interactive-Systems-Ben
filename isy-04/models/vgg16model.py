from keras.models import Sequential
from keras.layers import Dropout, Convolution2D, MaxPooling2D, Flatten, Dense, Activation, ZeroPadding2D
from keras.layers.normalization import BatchNormalization


class VGG16Model:
    img_rows = 28
    img_cols = 28

    @staticmethod
    def load_inputshape():
        return VGG16Model.img_rows, VGG16Model.img_cols, 1

    @staticmethod
    def reshape_input_data(x_train, x_test):
        x_train = x_train.reshape(x_train.shape[0], VGG16Model.img_rows, VGG16Model.img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], VGG16Model.img_rows, VGG16Model.img_cols, 1)
        return x_train, x_test


    @staticmethod
    def load_model(classes=10):
        # adapted from https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
        model = Sequential()
        model.add(Convolution2D(64, 3, activation='relu', input_shape=(VGG16Model.img_rows, VGG16Model.img_cols,1),padding='same'))
        # model.add(Convolution2D(64, 3, activation='relu',padding='same'))
        # model.add(MaxPooling2D((2, 2)))
        #
        # model.add(Convolution2D(128, 3, activation='relu',padding='same'))
        # model.add(Convolution2D(128, 3, activation='relu',padding='same'))
        # model.add(MaxPooling2D((2, 2)))
        #
        # model.add(Convolution2D(256, 3, activation='relu',padding='same'))
        # model.add(Convolution2D(256, 3, activation='relu',padding='same'))
        # model.add(Convolution2D(256, 3, activation='relu',padding='same'))
        # model.add(MaxPooling2D((2, 2)))
        #
        # model.add(Convolution2D(512, 3, activation='relu',padding='same'))
        # model.add(Convolution2D(512, 3, activation='relu',padding='same'))
        # model.add(Convolution2D(512, 3, activation='relu',padding='same'))
        # model.add(MaxPooling2D((2, 2)))
        #
        # # Images seem to be to small at the time of the maxpooling here
        # # model.add(Convolution2D(512, 3, activation='relu',padding='same'))
        # # model.add(Convolution2D(512, 3, activation='relu',padding='same'))
        # # model.add(Convolution2D(512, 3, activation='relu',padding='same'))
        # # model.add(MaxPooling2D((2, 2)))
        #
        # model.add(Flatten())
        # model.add(Dense(4096, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(4096, activation='relu'))
        # model.add(Dropout(0.5))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        # model.add(ZeroPadding2D((1, 1)))
        # model.add(Convolution2D(512, 3, 3, activation='relu'))
        # model.add(ZeroPadding2D((1, 1)))
        # model.add(Convolution2D(512, 3, 3, activation='relu'))
        # model.add(ZeroPadding2D((1, 1)))
        # model.add(Convolution2D(512, 3, 3, activation='relu'))
        # model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(classes, activation='softmax'))
        # model.add(BatchNormalization())
        model.compile(loss='categorical_crossentropy', optimizer ='adam', metrics = ['accuracy'])

        return model