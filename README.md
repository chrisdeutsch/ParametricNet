# ParametricNet

## How to setup the training environment

For example for a cvmfs-enabled CentOS7 device:
```bash
setupATLAS

# Check 'lsetup "lcgenv -p LCG_96b x86_64-centos7-gcc62-opt"' for available packages
lcg_reqs="Python
pip
numpy
pandas
scikitlearn
keras
tensorflow"

for req in ${lcg_reqs}; do
    echo "Setting up ${req}"
    lsetup "lcgenv -p LCG_96b x86_64-centos7-gcc62-opt ${req}" > /dev/null
done

# Not available in LGC env
echo "Setting up uproot"
python -m pip install --user uproot > /dev/null
```

## How to perform the training

Try looking at the output of `--help` of the `train.py` script.

A couple of important options are:

- `--fold {even,odd}`: specify whether to train on events with even / odd event numbers
- `--input-vars`: list of input variables used for the training (this does not include the parameter)
- `--sig-tree-regex`: Python regular expression to find signal trees and mass.
                      E.g. for a tree `Xtohh500` this would be `(Xtohh(\d+))` (note the two capture groups).
- `--bkg-trees`: list of tree names for background

**Important**: Ensure that your selection is already applied at the ntuple
stage. The only selection applied by the training script is removing negative
weights and selecting the fold.

For more options check the `DataLoader.__init__` and `ParametricNet.__init__`
functions and adapt the training script if necessary.


## How to convert model with `lwtnn`

```bash
# Check out ParametricNet (if you don't have it yet)
git clone ssh://git@gitlab.cern.ch:7999/cdeutsch/parametricnet.git

# Check out lwtnn
git clone git@github.com:lwtnn/lwtnn.git

# Set paths (adapt as necessary)
PATH="$(readlink -e parametricnet/scripts):${PATH}"
PATH="$(readlink -e lwtnn/scripts):$(readlink -e lwtnn/converters):${PATH}"

setupATLAS

# Set up python3 because lwtnn does not support python 2
lcg_reqs="Python
keras
tensorflow"

for req in ${lcg_reqs}; do
    echo "Setting up ${req}"
    lsetup "lcgenv -p LCG_96bpython3 x86_64-centos7-gcc62-opt ${req}" > /dev/null
done


lwtnn-split-keras-network.py model.h5
kerasfunc2json.py architecture.json weights.h5 > input_spec.json

fill_spec.py input_spec.json scaler.json > input_spec_filled.json

kerasfunc2json.py architecture.json weights.h5 input_spec_filled.json > nn.json
```

## How to use in Reader / MIA

```cpp
// Create the Parametric Network somewhere in 'initialize'
m_pnn = std::make_unique<ParametricNet>();

// Store your tuning in ParametricNet/data and load as follows:
const std::string workdir = gSystem->Getenv("WorkDir_DIR");
m_pnn->init(workdir + "/data/ParametricNet/pnet_even_v12.json",
            workdir + "/data/ParametricNet/pnet_odd_v12.json");

// The first argument to 'init' is the net trained on even, the second on odd event numbers
// When applying with 'evaluate(mass, Fold::Odd)' it applies the network that was trained
// on even numbers and vice versa

// In 'execute' set your variables once (the names are the same as in the training ntuple):
m_pnn->reset();
m_pnn->set_variable("dRTauTau", dRTauTau);
m_pnn->set_variable("dRBB", dRBB);
m_pnn->set_variable("mMMC", mMMC);
m_pnn->set_variable("mBB", mBB);
m_pnn->set_variable("mHH", mHH);

// And evaluate at your masses of interest:
for (const auto mass : masses) {
    using Fold = ParametricNet::Fold;
    const auto fold = m_event->eventNumber() % 2 == 0 ? Fold::Even : Fold::Odd;

    m_pnn->evaluate(mass, fold);
}
```
