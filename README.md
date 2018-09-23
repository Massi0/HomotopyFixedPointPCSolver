# Homotopy Continuation Solver

A homotopy method solver that finds fixed points of C2 functions. The solver is an implementation of the predictor-corrector scheme.

## Prerequisites

* scipy
* numpy
* pickle

## Getting Started

* "main.py" contains 3 test examples. 

* test 1: finding a root a multi-polynomial equation
* test 2: finding the equilibrium of an LSTM neural network map
* test 3: finding the equilibria of LSTM NN during the training
### To Run:

* Step 1: choose test ={1,2,3}.
* Step 2: run python main.py

## Orginization of the code

* HomPCSolver/*Impl_.py: abstract classes that contain the main implementation of the homotopy solver. Any personal use should inherent from the class "HomotopyPCSolver"

* demo/LSTMS.py: an implementation of a simple n dimensional LSTM map.

### Reference:
* *Garcia, C. B., & Zangwill, W. I. Pathways to solutions, fixed points, and equilibria. 1981.*