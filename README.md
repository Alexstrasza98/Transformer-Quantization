# Transformer-Quantization
The final project repository for 2022 Spring COMS6998-009 Deep Learning System Performance in Columbia University.

# Simple Code Structure
`figures`: figures included in report

`quantization`: core codes including transformer definition, quantization, binarization
 
`res`: models, training logs, test results of different experiments

`utils`: utils functions

`main.ipynb`: main training codes (change to .py with argparser later)

`plot.ipynb`: plotting codes (change to .py later)

# Project Schedule
Accomplished:
1. Define baseline model structure and training configurations
2. Test binarization on different parts of transformer to see sensitivity of each part

Expected:
1. Try to only quant Q,K part of MHA, see if any improvements
