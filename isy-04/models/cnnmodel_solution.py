from keras.models import Sequential
from keras.layers import Dropout, Convolution2D, MaxPooling2D, Flatten, Dense, Activation


class CNNModel:
    img_rows = 28
    img_cols = 28

    @staticmethod
    def load_inputshape():
        return CNNModel.img_rows, CNNModel.img_cols, 1

    @staticmethod
    def reshape_input_data(x_train, x_test):
        x_train = x_train.reshape(x_train.shape[0], CNNModel.img_rows, CNNModel.img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], CNNModel.img_rows, CNNModel.img_cols, 1)
        return x_train, x_test


    @staticmethod
    def load_model(classes=10):
        model = Sequential()
        model.add(Convolution2D(32, (3, 3), padding='same', activation='relu',
                                input_shape=(CNNModel.img_rows, CNNModel.img_cols, 1)))
        model.add(Convolution2D(32, 1, activation='relu'))
        model.add(MaxPooling2D())
        model.add(Dropout(0.25))
        model.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Convolution2D(64, 1, activation='relu'))
        model.add(MaxPooling2D())
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(units=512, activation='relu'))
        model.add(Dense(units=classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model
    # def load_model(classes=10):
    #     model = Sequential()
    #     model.add(Convolution2D(32, (3, 3), padding='same', input_shape=CNNModel.load_inputshape(), activation='relu'))
    #     model.add(Convolution2D(32, (3, 3), activation='relu'))
    #     model.add(MaxPooling2D(pool_size=(2, 2)))
    #     model.add(Dropout(0.25))
    #
    #     model.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
    #     model.add(Convolution2D(64, (3, 3), activation='relu'))
    #     model.add(MaxPooling2D(pool_size=(2, 2)))
    #     model.add(Dropout(0.25))
    #
    #     model.add(Flatten())
    #     model.add(Dense(512, activation='relu'))
    #     model.add(Dropout(0.5))
    #     model.add(Dense(classes))
    #     model.add(Activation('softmax'))
    #
    #     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #     return model
