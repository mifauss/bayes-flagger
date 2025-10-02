Code for _Bayesian Selection Policies for Human-in-the-Loop Anomaly Detectors with Applications in Test Security_
=====================================

This repository contains Python code and data to re-generate the numerical results in the paper _Bayesian Selection Policies for Human-in-the-Loop Anomaly Detectors with Applications in Test Security_.  

## Getting Started

1. Install required packages, for example via `pip`:
```python
 pip install -r requirements.txt
```

2. Use [Jupyter](https://jupyter.org/) or [IPython](https://ipython.org/) to run the example notebooks. For a simple example that demonstrates how to use the Bayesian flagger see the  `example_basics.ipynb` notebook.

## Organization

- `flagger.py` implements the Bayes flagger proposed in the paper
- `flagger_tools.py` implements helper classes and functions to run simulations
- `example_basics.ipynb` demonstrates how to initialize and use the Bayes flagger
- `example_synthetic_proposed.ipynb` evaluates the proposed flagger on synthetic data
- `example_synthetic_baseline.ipynb` evaluates baseline flaggers on synthetic data
- `example_real_world_proposed.ipynb` evaluates the proposed flagger on real world data
- `example_real_world_baseline.ipynb` evaluates baseline flaggers on real world data

See the paper for more details on both simulation setups.
