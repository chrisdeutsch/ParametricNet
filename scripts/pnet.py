import re

import numpy as np
import pandas as pd

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
        print("Current LR: " + str(K.eval(lr_with_decay)))


def get_sig_df(infile):
    # Bytes to string
    trees = [key.decode() for key in infile.keys()]

    # Mapping mass to tree
    mass_map = {}

    sig_pattern = re.compile(r"(Xtohh(\d+))")
    for tree in trees:
        match = sig_pattern.search(tree)
        if match:
            treename, mass = match.groups()
            mass_map[int(mass)] = treename

    # Read dataframes
    dfs = []
    for mass, tree in mass_map.items():
        df = infile[tree].pandas.df()
        df["mass"] = mass
        df["mass_scaled"] = mass_transform(mass)
        df["signal"] = 1
        dfs.append(df)

    # Return merged dataframes
    return pd.concat(dfs)



def get_bkg_df(infile):
    bkg_trees = ["ttbar", "stop", "Ztautau", "Fake", "VH", "Diboson", "Wtaunu"]

    # Read dataframes
    dfs = []
    for tree in bkg_trees:
        df = infile[tree].pandas.df()
        df["mass"] = -1
        df["mass_scaled"] = -1.
        df["signal"] = 0
        dfs.append(df)

    # Return merged dataframes
    return pd.concat(dfs)


def apply_selection(df, fold="even"):
    sel_pos_weight = df["weight"] > 0

    if fold == "even":
        sel_fold = df["event_number"] % 2 == 0
    elif fold == "odd":
        sel_fold = df["event_number"] % 2 == 1
    else:
        raise RuntimeError("Wrong fold specified.")

    return df[sel_pos_weight & sel_fold].copy()


def equalize_signal_weight(sig_df, target_weight=10000.0):
    masspoints = sig_df.mass.unique()
    num_masspoints = len(masspoints)
    weight_per_masspoint = target_weight / num_masspoints

    scale_map = {}
    for mass in masspoints:
        weight_sum = sig_df.loc[sig_df.mass == mass, "weight"].sum()
        scale_map[mass] = weight_per_masspoint / weight_sum

    sig_df["weight_scaled"] = sig_df["weight"] * sig_df["mass"].map(scale_map)


def mass_transform(mass):
    mass_min = 251.0
    mass_max = 1000.0
    return (mass - mass_min) / (mass_max - mass_min)


def get_model(n_invars, n_hidden_nodes):
    if not isinstance(n_hidden_nodes, list):
        n_hidden_nodes = [n_hidden_nodes]

    x = Input(shape=(n_invars,))
    d = x
    for n in n_hidden_nodes:
        d = Dense(n, activation="relu") (d)

    y = Dense(1, activation="sigmoid")(d)
    return Model(x, y)


def sample_bkg_mass(mass_arr, y):
    masspoints = np.unique(mass_arr[y == 1])
    sampled_mass = np.random.choice(masspoints, size=np.count_nonzero(y == 0))
    mass_arr[y == 0] = sampled_mass


def prepare_inputs(sig_df, bkg_df):
    # Total background weight
    bkg_weight = bkg_df["weight"].sum()

    # Equalize signal weights (total sig weight should match bkg)
    equalize_signal_weight(sig_df, target_weight=bkg_weight)

    # Equalize signal / background weights
    bkg_df["weight_scaled"] = bkg_df["weight"]

    # Check if reweighting worked
    sig_weight = sig_df.weight_scaled.sum()
    bkg_weight = bkg_df.weight_scaled.sum()
    assert np.isclose(sig_weight, bkg_weight)

    # Combine sig and bkg dataframe
    df = pd.concat([sig_df, bkg_df])

    # Arrays for training
    mva_vars = ["dRTauTau", "dRBB", "mMMC", "mBB", "mHH", "mass_scaled"]
    signal_var = "signal"
    weight_var = "weight_scaled"

    X = df[mva_vars].values
    Y = df[signal_var].values.astype(int)
    W = df[weight_var].values

    X, Y, W = shuffle(X, Y, W)

    return X, Y, W


def train(X, Y, W, **kwargs):
    n_epochs = kwargs.setdefault("epochs", 50)
    batch_size = kwargs.setdefault("batch_size", 64)
    layer_size = kwargs.setdefault("layer_size", [32, 32, 32])
    learning_rate = kwargs.setdefault("learning_rate", 0.1)
    learning_rate_decay = kwargs.setdefault("learning_rate_decay", 1e-5)

    # Apply preprocessing
    scaler = RobustScaler()
    X[:, :-1] = scaler.fit_transform(X[:, :-1])

    # Training of the PNN
    _, n_invars = X.shape
    model = get_model(n_invars, layer_size)
    model.summary()

    lr_printer = LRPrinter()
    sgd = SGD(lr=learning_rate, momentum=0.9, nesterov=True, decay=learning_rate_decay)
    model.compile(optimizer=sgd, loss="binary_crossentropy", metrics=["accuracy"])

    for epoch in range(n_epochs):
        X, Y, W = shuffle(X, Y, W)

        # Sample bkg masses
        sample_bkg_mass(X[:, -1], Y)

        model.fit(X, Y, sample_weight=W, batch_size=batch_size, epochs=1, verbose=1, callbacks=[lr_printer])

    return scaler, model
