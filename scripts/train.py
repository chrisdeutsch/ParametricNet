#!/usr/bin/env python
import argparse
import joblib
import json
import logging

from dataloader import DataLoader
from parametricnet import ParametricNet

logging.getLogger().setLevel(logging.INFO)


def main(args):
    # Load inputs & preprocess
    data_loader = DataLoader(fold=args.fold)
    data_loader.bkg_trees = args.bkg_trees
    data_loader.input_vars = args.input_vars

    X, Y, W = data_loader.load(args.ntuple)

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
    parser.add_argument("--bkg-trees", default=["ttbar", "stop", "Ztautau",
                                                "Fake", "VH", "Diboson",
                                                "Wtaunu"])
    parser.add_argument("--input-vars", default=["dRTauTau", "dRBB", "mMMC",
                                                 "mBB", "mHH"])
    parser.add_argument("--fold", choices=["even", "odd"], required=True)

    # ParametricNet parameters
    parser.add_argument("-e", "--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--learning-rate", default=0.1, type=float)
    parser.add_argument("--learning-rate-decay", default=1e-5, type=float)
    parser.add_argument("-l", "--layer-size", default=[32, 32, 32], nargs="+",
                        type=int, help="List of hidden layer sizes")

    args = parser.parse_args()
    main(args)
