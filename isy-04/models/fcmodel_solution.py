from keras.models import Sequential
from keras.layers import Dropout, Dense, Activation


class FCModel:

    @staticmethod
    def load_inputshape():
        return (784,)

    @staticmethod
    def reshape_input_data(x_train, x_test):
        return x_train, x_test

    @staticmethod
    def load_model(classes=10):
        model = Sequential()
        model.add(Dense(512, input_shape=FCModel.load_inputshape()))
        model.add(Activation('relu'))  # An "activation" is just a non-linear function applied to the output
        # of the layer above. Here, with a "rectified linear unit",
        # we clamp all values below 0 to 0.

        model.add(Dropout(0.2))  # Dropout helps protect the model from memorizing or "overfitting" the training data
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(classes))
        model.add(Activation('softmax'))  # This special "softmax" activation among other things,
        # ensures the output is a valid probaility distribution, that is
        # that its values are all non-negative and sum to 1.
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
