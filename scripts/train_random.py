#!/usr/bin/env python
import argparse
import random
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("seed", type=int)
parser.add_argument("ntup")
args = parser.parse_args()

random.seed(args.seed)

epochs = random.choice([50, 100, 200, 400])
batch_size = random.choice([64, 128, 256])
learning_rate = random.choice([0.01, 0.02, 0.05, 0.1, 0.2])
learning_rate_decay = random.choice([1e-6, 1e-5, 1e-4, 1e-3])

n_layers = random.randint(1, 5)

# Possible layer sizes
layer_sizes = [16, 32, 64, 128]

# Some code to generate reasonable layouts
top_layer = random.choice(layer_sizes)
mid_layer = random.choice(layer_sizes)
bot_layer = random.choice(layer_sizes)

while top_layer > mid_layer:
    top_layer = random.choice(layer_sizes)

while bot_layer > mid_layer:
    bot_layer = random.choice(layer_sizes)

layers = []
for i in range(n_layers):
    if i == 0:
        layers.append(top_layer)
    elif i == n_layers - 1:
        layers.append(bot_layer)
    else:
        layers.append(mid_layer)

layers = [str(l) for l in layers]

arglist = []
arglist += ["-e", str(epochs)]
arglist += ["--batch-size", str(batch_size)]
arglist += ["--learning-rate", str(learning_rate)]
arglist += ["--learning-rate-decay", str(learning_rate_decay)]
arglist += ["--layer-size"] + layers


# Even-fold
with open("train_even.log", "w") as f:
    f.write("Arguments: " + repr(arglist) + "\n")
    f.flush()
    train_even = subprocess.Popen(["train.py", args.ntup,
                                   "--fold", "even",
                                   "--cv", "5"]
                                  + arglist,
                                  stdout=f)
    train_even.communicate()
    if train_even.returncode:
        sys.exit(train_even.returncode)

# Odd-fold
with open("train_odd.log", "w") as f:
    f.write("Arguments: " + repr(arglist) + "\n")
    f.flush()
    train_odd = subprocess.Popen(["train.py", args.ntup,
                                   "--fold", "odd",
                                  "--cv", "5"]
                                  + arglist,
                                  stdout=f)
    train_odd.communicate()
    if train_odd.returncode:
        sys.exit(train_odd.returncode)
