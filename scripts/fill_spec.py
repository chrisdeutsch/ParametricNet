#!/usr/bin/env python
import argparse
import json

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("spec")
parser.add_argument("scaler")

args = parser.parse_args()


# Load network specification
with open(args.spec) as f_spec:
    arch = json.load(f_spec)

# Load scaler specification
with open(args.scaler) as f_scaler:
    scaler = json.load(f_scaler)


# Transforming from sklearn to lwtnn representation
input_vars = scaler["input_vars"]
offsets = -np.array(scaler["center"])
scales = 1.0 / np.array(scaler["scale"])

assert len(input_vars) == len(offsets)
assert len(offsets) == len(scales)

# Enter in architecture specification
inputs, = spec["inputs"]
outputs, = spec["outputs"]

inputs["name"] = "input_layer"

for i, var in enumerate(inputs["variables"]):
    var["name"] = input_vars[i]
    var["offset"] = offsets[i]
    var["scale"] = scales[i]

outputs["name"] = "output_layer"
outputs["labels"][0] = "sig_prob"

print(json.dumps(spec, indent=4, sort_keys=True))
