#!/usr/bin/env python
import argparse
import pnet

import uproot
import numpy as np
from hyperopt import hp, fmin, tpe


from sklearn.model_selection import StratifiedKFold


def evaluate(X, Y, W, scaler, model):
    X[:, :-1] = scaler.transform(X[:, :-1])
    metrics = model.evaluate(X, Y, sample_weight=W, verbose=0)

    loss, acc = metrics

    return loss


def do_cv(X_even, Y_even, W_even, X_odd, Y_odd, W_odd, **kwargs):
    # Cross validation loop
    results = []
    skf = StratifiedKFold(n_splits=5)
    for X, Y, W in [(X_even, Y_even, W_even), (X_odd, Y_odd, W_odd)]:
        # CV Loop
        for train_idx, test_idx in skf.split(X, Y):
            scaler, model = pnet.train(X[train_idx], Y[train_idx], W[train_idx], **kwargs)

            # Test training
            X_test, Y_test, W_test = X[test_idx], Y[test_idx], W[test_idx]
            pnet.sample_bkg_mass(X_test[:, -1], Y_test)
            results.append(evaluate(X_test, Y_test, W_test, scaler, model))

    return np.mean(results)


if __name__ == "__main__":
    f = uproot.open("mva_ntup.root")

    sig_df = pnet.get_sig_df(f)
    bkg_df = pnet.get_bkg_df(f)

    sig_df_even = pnet.apply_selection(sig_df, fold="even")
    bkg_df_even = pnet.apply_selection(bkg_df, fold="even")

    sig_df_odd = pnet.apply_selection(sig_df, fold="odd")
    bkg_df_odd = pnet.apply_selection(bkg_df, fold="odd")

    X_even, Y_even, W_even = pnet.prepare_inputs(sig_df_even, bkg_df_even)
    X_odd, Y_odd, W_odd = pnet.prepare_inputs(sig_df_odd, bkg_df_odd)


    space = {
        "learning_rate": hp.lognormal("lr", -2, 1),
        "learning_rate_decay": hp.loguniform("lr_decay", np.log(1e-9), np.log(1e-4)),
        "layer_number": hp.quniform("layer_number", 1, 4, 1),
        "layer_size": 1 + hp.quniform("layer_size", 8, 128, 2)
    }


    def objective(args):
        print(args)
        args["layer_number"] = int(args["layer_number"])
        args["layer_size"] = int(args["layer_size"])

        return do_cv(X_even, Y_even, W_even, X_odd, Y_odd, W_odd, epochs=100, **args)

    best = fmin(objective, space, algo=tpe.suggest, max_evals=50)

    print(best)


