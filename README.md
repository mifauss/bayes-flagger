Code for _Bayesian Selection Policies for Human-in-the-Loop Anomaly Detectors_
=====================================

This repository contains Python code and data to re-generate the numerical results in the paper _Bayesian Selection Policies for Human-in-the-Loop Anomaly Detectors_.  

## Getting Started

1. Install required packages, for example via `pip`:
```python
 pip install -r requirements.txt
```

2. Use [Jupyter](https://jupyter.org/) or [IPython](https://ipython.org/) to run the example notebooks.

## Organization

- `flagger.py` implements the Bayes flagger proposed in the paper
- `flagger_tools.py` implements helper classes and functions to run simulations
- `example_synthetic.ipynb` evaluates the flagger on synthetic data
- `example_real_world.ipynb` evaluates the flagger on real world data

See the paper for more details on both simulation setups.