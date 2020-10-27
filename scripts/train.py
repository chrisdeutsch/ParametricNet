#!/usr/bin/env python
import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

from dataloader import DataLoader
from parametricnet import ParametricNet
from evaluation import evaluate
from sklearn.model_selection import StratifiedKFold

logging.getLogger().setLevel(logging.INFO)


def main(args):
    # Load inputs & preprocess
    data_loader = DataLoader(fold=args.fold)
    data_loader.sig_tree_regex = args.sig_tree_regex
    data_loader.bkg_trees = args.bkg_trees
    data_loader.input_vars = args.input_vars
    data_loader.weight_name = args.weight_name
    data_loader.event_number_variable = args.event_number_variable

    X, Y, W = data_loader.load(args.ntuple)

    # K-fold cross validation loop
    if args.cv:
        results = []

        print("Total background: " + repr(W[Y == 0].sum()))
        print("Total signal: " + repr(W[Y == 1].sum()))

        # K-fold over truth-mass to get similar distributions in every split
        skf = StratifiedKFold(n_splits=args.cv)
        _, target = np.unique(X[:, -1], return_inverse=True)
        for train_idx, test_idx in skf.split(np.zeros_like(target), target):
            net = ParametricNet()
            net.epochs = args.epochs
            net.batch_size = args.batch_size
            net.learning_rate = args.learning_rate
            net.learning_rate_decay = args.learning_rate_decay
            net.layer_size = args.layer_size

            X_train, Y_train, W_train = X[train_idx], Y[train_idx], W[train_idx]
            net.train(X_train, Y_train, W_train)

            X_test, Y_test, W_test = X[test_idx], Y[test_idx], W[test_idx]
            results.append(evaluate(X_test, Y_test, W_test, data_loader, net))

        pd.DataFrame(results).to_csv("cv_results_{}.csv".format(args.fold))
        sys.exit(0)


    # Training
    net = ParametricNet()
    net.epochs = args.epochs
    net.batch_size = args.batch_size
    net.learning_rate = args.learning_rate
    net.learning_rate_decay = args.learning_rate_decay
    net.layer_size = args.layer_size

    net.train(X, Y, W)

    # Save outputs
    scaler_out = "scaler.json"
    logging.info("Saving scaling factors in " + scaler_out)

    # Fill variable names & scaling factors into json
    outputs = {
        "input_vars": data_loader.input_vars + ["mass"],
        "center": list(net.scaler.center_) + [data_loader.mass_min],
        "scale": list(net.scaler.scale_) + [data_loader.mass_max - data_loader.mass_min]
    }

    with open(scaler_out, "w") as outf:
        json.dump(outputs, outf, indent=4)

    model_out = "model.h5"
    logging.info("Saving network weights in " + model_out)
    net.model.save(model_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ntuple", help="Ntuple with MVA trees")

    # DataLoader parameters
    parser.add_argument("--sig-tree-regex", default=r"(Xtohh(\d+))")
    parser.add_argument("--bkg-trees", nargs="+",
                        default=["ttbar", "ttbarFakesMC", "Ztautau", "Fake", "VH",
                                 "Htautau", "ttH", "Wtaunu", "Diboson", "singletop"])
    parser.add_argument("--input-vars", nargs="+",
                        default=["dRTauTau", "dRBB", "mMMC", "mBB", "mHH"])
    parser.add_argument("--weight-name", default="weight")
    parser.add_argument("--event-number-variable", default="event_number")
    parser.add_argument("--fold", choices=["even", "odd"], required=True)

    # ParametricNet parameters
    parser.add_argument("-e", "--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--learning-rate", default=0.1, type=float)
    parser.add_argument("--learning-rate-decay", default=1e-5, type=float)
    parser.add_argument("-l", "--layer-size", default=[32, 32, 32], nargs="+",
                        type=int, help="List of hidden layer sizes")

    # Cross validation
    parser.add_argument("--cv", type=int, default=None)

    args = parser.parse_args()
    main(args)
