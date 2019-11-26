#!/usr/bin/env python
import argparse
import joblib
import pnet

import uproot



def main(args):
    # Load inputs
    f = uproot.open(args.ntuple)
    sig_df = pnet.get_sig_df(f)
    bkg_df = pnet.get_bkg_df(f)

    # To remove signals from df
    if args.remove_mass:
        sig_df = sig_df[sig_df.mass != args.remove_mass]

    # Apply selection
    if args.even:
        fold = "even"
    elif args.odd:
        fold = "odd"
    else:
        raise RuntimeError("No fold specified")

    sel_sig_df = pnet.apply_selection(sig_df, fold=fold)
    sel_bkg_df = pnet.apply_selection(bkg_df, fold=fold)

    X, Y, W = pnet.prepare_inputs(sel_sig_df, sel_bkg_df)

    scaler, model = pnet.train(X, Y, W,
                               epochs=args.epochs,
                               layer_size=args.layer_size,
                               learning_rate=args.learning_rate,
                               learning_rate_decay=args.learning_rate_decay
    )
    joblib.dump(scaler, "scaler.pkg")
    model.save("model.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ntuple", help="Ntuple with MVA trees")
    parser.add_argument("-l", "--layer-size", default=[32, 32, 32], nargs="+",
                        type=int, help="List of hidden layer sizes")
    parser.add_argument("-e", "--epochs", default=100, type=int)
    parser.add_argument("--learning-rate", default=0.1, type=float)
    parser.add_argument("--learning-rate-decay", default=1e-5, type=float)

    parser.add_argument("--remove-mass", type=int, default=None)

    split = parser.add_mutually_exclusive_group(required=True)
    split.add_argument("--even", action="store_true",
                       help="Run on even event numbers")
    split.add_argument("--odd", action="store_true",
                       help="Run on odd event numbers")

    args = parser.parse_args()

    main(args)
