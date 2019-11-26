# ParametricNet


## How to use in your code

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

// In 'execute' set your variables once:
m_pnn->set_vars(dRTauTau, dRBB, mMMC, mBB, mHH);

// And evaluate at your masses of interest:
for (const auto mass : masses) {
    using Fold = ParametricNet::Fold;
    const auto fold = m_event->eventNumber() % 2 == 0 ? Fold::Even : Fold::Odd;
    
    m_pnn->evaluate(mass, fold);
}
```

## How to convert model with `lwtnn`

```bash
# Activate your python environment that has keras (python3)

git clone git@github.com:lwtnn/lwtnn.git

PATH="$(readlink -e lwtnn/scripts):$(readlink -e lwtnn/converters):${PATH}"

lwtnn-split-keras-network.py model.h5
kerasfunc2json.py architecture.json weights.h5 > input_spec.json

fill_spec.py input_spec.json scaler.pkg > input_spec_filled.json

kerasfunc2json.py architecture.json weights.h5 input_spec_filled.json > nn.json
```
