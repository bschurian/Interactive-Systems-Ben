from keras.models import Sequential
from keras.layers import Dropout, Convolution2D, MaxPooling2D, Flatten, Dense, Activation, ZeroPadding2D
from keras.layers.normalization import BatchNormalization


class AlexNetModel:
    img_rows = 28
    img_cols = 28

    @staticmethod
    def load_inputshape():
        return AlexNetModel.img_rows, AlexNetModel.img_cols, 1

    @staticmethod
    def reshape_input_data(x_train, x_test):
        x_train = x_train.reshape(x_train.shape[0], AlexNetModel.img_rows, AlexNetModel.img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], AlexNetModel.img_rows, AlexNetModel.img_cols, 1)
        return x_train, x_test


    @staticmethod
    def load_model(classes=10):
        # adapted from by https://github.com/eweill/keras-deepcv/blob/master/models/classification/model.py
        model = Sequential()
        model.add(Convolution2D(96, (11, 11),padding='same',activation='relu',input_shape=(AlexNetModel.img_rows, AlexNetModel.img_cols, 1)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(256, (5, 5), padding='same',activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, (3, 3), padding='same',activation='relu'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(1024, (3, 3), padding='same',activation='relu'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(1024, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(3072))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))
        model.add(BatchNormalization())
        model.compile(loss='categorical_crossentropy', optimizer ='adam', metrics = ['accuracy'])

        return model