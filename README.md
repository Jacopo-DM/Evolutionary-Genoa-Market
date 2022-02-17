# Evolutionary Genoa

## Environment

Install the correct packages using (possibly in dedicated env):

```
pip install -r requirements.txt
```

## Various Price Paths & Log Returns
To recreate the results of the analysis on different price paths run:

```
python plots.py
```

## Sensitivity Analysis (WARNING: Takes Days!)

Perform sensitivity analysis on Evo Genoa run:

```
python sensitivity.py
```

## Directory Structure

./
├── README.md (this file)
├── EvoMarket.py:   The new model we made
├── EvoTrader.py:   Actor for model we made
├── analysis.py:    File containing various analysis
├── Stats.py:       Runs tests from 'analysis.py' in one go
├── data.py:        Various data fetching functions
├── intersect.py:   Calculate intersection between curves (external code)
├── montecarlo.py:  Simulation code for Black Sholes and Stochastic
├── plots.py:       Various plotting functions
├── sensitivity.py: Perform sensitivity analysis (code to run for sensitivity)
└── utils.py:       Some useful functions that don't fit in a box
├── data:           Output directory for 'data.py'
├── plots:          Output directory for 'plots.py' (main code to run)
├── requirements.txt