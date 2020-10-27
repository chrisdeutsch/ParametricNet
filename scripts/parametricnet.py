import logging

import numpy as np

from sklearn.utils import shuffle
from sklearn.preprocessing import RobustScaler

from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import SGD
from keras.callbacks import Callback
from keras import backend as K


class LRPrinter(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print("Current learning rate: " + str(K.eval(lr_with_decay)))


class ParametricNet:
    def __init__(self, **kwargs):
        # Optimization settings
        self.epochs = kwargs.setdefault("epochs", 100)
        self.batch_size = kwargs.setdefault("batch_size", 64)
        self.learning_rate = kwargs.setdefault("learning_rate", 0.1)
        self.learning_rate_decay = kwargs.setdefault("learning_rate_decay", 1e-5)
        self.momentum = kwargs.setdefault("momentum", 0.9)
        self.nesterov = kwargs.setdefault("nesterov", True)

        # Architecture settings
        self.layer_size = kwargs.setdefault("layer_size", [32, 32, 32])
        self.activation = kwargs.setdefault("activation", "relu")


    def get_model(self, n_invars):
        x = Input(shape=(n_invars,))
        d = x
        for n in self.layer_size:
            d = Dense(n, activation=self.activation)(d)

        y = Dense(1, activation="sigmoid")(d)
        return Model(x, y)


    def sample_parameter(self, param, Y):
        masspoints = np.unique(param[Y == 1])
        sampled_mass = np.random.choice(masspoints, size=np.count_nonzero(Y == 0))
        param[Y == 0] = sampled_mass


    def train(self, X, Y, W):
        # Apply preprocessing
        # Scale everything except for last column which is the parameter
        self.scaler = RobustScaler()
        X[:, :-1] = self.scaler.fit_transform(X[:, :-1])

        # Keras model
        _, n_invars = X.shape
        self.model = self.get_model(n_invars)
        self.model.summary()

        # To print the learning rate after every epoch
        lr_printer = LRPrinter()

        # Optimizer
        sgd = SGD(lr=self.learning_rate, momentum=self.momentum,
                  nesterov=self.nesterov, decay=self.learning_rate_decay)

        self.model.compile(optimizer=sgd, loss="binary_crossentropy", metrics=["accuracy"])

        # Epoch loop
        for epoch in range(self.epochs):
            X, Y, W = shuffle(X, Y, W)

            # Sample parameter
            self.sample_parameter(X[:, -1], Y)

            self.model.fit(X, Y, sample_weight=W, batch_size=self.batch_size,
                           epochs=1, verbose=2, callbacks=[lr_printer])


    def evaluate(self, X, param):
        X_copy = np.copy(X)
        X_copy[:, :-1] = self.scaler.transform(X[:, :-1])
        X_copy[:, -1] = param
        return self.model.predict(X_copy)
