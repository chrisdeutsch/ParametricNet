import re
import logging

import uproot
import numpy as np
import pandas as pd
from sklearn.utils import shuffle


class DataLoader:
    def __init__(self, **kwargs):
        # Configuration of input trees
        self.sig_tree_regex = kwargs.setdefault("sig_tree_regex", r"(Xtohh(\d+))")
        self.bkg_trees = kwargs.setdefault(
            "bkg_trees", ["ttbar", "stop", "Ztautau", "Fake", "VH", "Diboson",
                          "Wtaunu"])

        # Input variable names (as in tree)
        self.input_vars = kwargs.setdefault(
            "input_vars", ["dRTauTau", "dRBB", "mMMC", "mBB", "mHH"])

        # Weight name
        self.weight_name = kwargs.setdefault("weight_name", "weight")

        # Variable storing the event number
        self.event_number_variable = kwargs.setdefault("event_number_variable", "event_number")

        # Selection
        self.fold = kwargs.setdefault("fold", "even")
        assert self.fold == "even" or self.fold == "odd"


    def signal_df(self, infile):
        # Trees are saved as bytes
        trees = [key.decode() for key in infile.keys()]

        # Mapping mass to tree
        mass_map = {}

        # Find signal trees according to regex
        logging.info("Using signal regex: " + self.sig_tree_regex)
        pattern = re.compile(self.sig_tree_regex)
        for tree in trees:
            match = pattern.search(tree)
            if not match:
                continue

            treename, mass = match.groups()
            mass_map[int(mass)] = treename
            logging.info("Adding signal tree {} with mass {}".format(treename, mass))

        # Setting mass transform
        self.mass_min = float(min(mass_map.keys()))
        self.mass_max = float(max(mass_map.keys()))
        self.mass_transform = lambda x: (x - self.mass_min) / (self.mass_max - self.mass_min)

        logging.info("Using mass transform (m - {mass_min}) / ({mass_max} - {mass_min})".format(
            mass_min=self.mass_min, mass_max=self.mass_max))

        assert abs(self.mass_transform(self.mass_min)) < 1e-9
        assert abs(self.mass_transform(self.mass_max) - 1.) < 1e-9

        # Read dataframes
        dfs = []
        for mass, tree in mass_map.items():
            df = infile[tree].pandas.df()
            df["mass"] = mass
            df["mass_scaled"] = self.mass_transform(mass)
            df["signal"] = 1
            dfs.append(df)

        return pd.concat(dfs)


    def background_df(self, infile):
        dfs = []
        for tree in self.bkg_trees:
            df = infile[tree].pandas.df()
            df["mass"] = -1
            df["mass_scaled"] = -1.
            df["signal"] = 0
            dfs.append(df)
            logging.info("Adding background tree " + tree)

        return pd.concat(dfs)


    def apply_selection(self, df):
        sel_pos_weight = df[self.weight_name] > 0

        if self.fold == "even":
            sel_fold = df[self.event_number_variable] % 2 == 0
        elif self.fold == "odd":
            sel_fold = df[self.event_number_variable] % 2 == 1
        else:
            raise RuntimeError("Error in fold specification")

        return df[sel_pos_weight & sel_fold].copy()


    def reweighting(self, sig_df, bkg_df):
        # Total background weight
        bkg_weight = bkg_df[self.weight_name].sum()

        # Equalize signal weights (total sig weight should match bkg)
        masspoints = sig_df.mass.unique()
        num_masspoints = len(masspoints)
        weight_per_masspoint = bkg_weight / num_masspoints

        scale_map = {}
        for mass in masspoints:
            weight_sum = sig_df.loc[sig_df.mass == mass, self.weight_name].sum()
            scale_map[mass] = weight_per_masspoint / weight_sum

            logging.info("Reweighting signal mass {} with factor {}".format(mass, scale_map[mass]))

        bkg_df["weight_scaled"] = bkg_df[self.weight_name]
        sig_df["weight_scaled"] = sig_df[self.weight_name] * sig_df["mass"].map(scale_map)

        # Sanity check
        assert abs(sig_df["weight_scaled"].sum() - bkg_df["weight_scaled"].sum()) < 1e-3


    def load(self, infile):
        logging.info("Loading " + infile)
        f = uproot.open(infile)

        sig_df = self.signal_df(f)
        bkg_df = self.background_df(f)

        logging.info("Applying selection for fold: " + self.fold)
        sig_df = self.apply_selection(sig_df)
        bkg_df = self.apply_selection(bkg_df)

        logging.info("Reweighting signal so that the integral matches background")
        self.reweighting(sig_df, bkg_df)

        # Combine for training
        df = pd.concat([sig_df, bkg_df])

        # Arrays for training
        X = df[self.input_vars + ["mass_scaled"]].values
        Y = df["signal"].values.astype(int)
        W = df["weight_scaled"].values

        X, Y, W = shuffle(X, Y, W)


        return X, Y, W
