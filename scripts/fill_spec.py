#!/usr/bin/env python
import argparse
import json
import joblib

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("spec")
parser.add_argument("scaler")

args = parser.parse_args()


# Load network specification
with open(args.spec) as f_spec:
    spec = json.load(f_spec)

# Load scaler
scaler = joblib.load(args.scaler)
offsets = -scaler.center_
scales = 1.0 / scaler.scale_

# Add scaled mass
offsets = np.hstack([offsets, [-251.0]])
scales = np.hstack([scales, [1.0 / (1000 - 251)]])

inputs, = spec["inputs"]
outputs, = spec["outputs"]

inputs["name"] = "input_layer"

variable_names = ["DRTauTau", "dRBB", "mMMC", "mBB", "mHH", "mass"]
assert len(variable_names) == len(offsets)

for i, var in enumerate(inputs["variables"]):
    var["name"] = variable_names[i]
    var["offset"] = offsets[i]
    var["scale"] = scales[i]


outputs["name"] = "output_layer"
outputs["labels"][0] = "sig_prob"



print(json.dumps(spec, indent=4, sort_keys=True))
