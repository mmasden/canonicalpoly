# Algorithmic Determination of the Combinatorial Structure of the Linear Regions of ReLU Neural Networks

This repository is the official implementation of _Algorithmic Determination of the Combinatorial Structure of the Linear Regions of ReLU Neural Networks_.

The included code computes the polyhedral complex of a ReLU Neural Network in Pytorch by computing only the vertices and their sign sequences. This allows for computation of topological invariants of subcomplexes of the canonical polyhedral complex, for example, its decision boundary. 

![torus](https://user-images.githubusercontent.com/38443979/169712774-31db512e-1e8b-4e00-b8fc-02d6bf4d3d0f.png)

## Requirements

To install requirements for obtaining the polyhedral decomposition of input space,run the following in a Python 3.9+ virtual environment.

```setup
pip install -r requirements.txt
```
If this conflicts with system properties, you may instead install the following in Python 3.9: 

* pytorch 1.11 by following the instructions [here](https://pytorch.org/get-started/locally/)
* matplotlib, 
* jupyter-notebook, and 
* numpy

The sample code is currently configured to run without requiring GPU support. 

For obtaining the topological decomposition of input space, we use Sage 9.0, with installation instructions provided [here](https://doc.sagemath.org/html/en/installation/index.html). No additional requirements are necessary.

## Obtaining Polyhedral Complexes

To obtain the polyhedral complexes for random initializations of neural networks, run:

```polyhedral complex
python3 Compute_Complexes_Initialization.py input_dimension hidden_layers minwidth maxwidth width_step n_trials 
```
For example, the command

```python3 Compute_Complexes_Initialization.py 2 2 6 12 3 20 ```  

will randomly initialize 20 neural networks for each architecture ```(2,n,n,1)``` (two hidden layers)
for values of n from 6 to 12 which are multiples of 3, and obtain the polyhedral complex for each of these networks.

The saved file is a Numpy .npz file for compatibility with Sage. It contains: 

* "complexes" (the sign sequences of all the vertices present in the initialized networks) 
* "points" (the location of all vertices present in the initialized networks) 
* "times" (the amount of time taken to compute all trials for each architecture) 
* "archs" (a record of the network architectures which were randomly initialized)


## Obtaining Topological Data

To obtain the Betti numbers of the resulting one-point compactified decision boundary, 
run the following **(outside of the virtual environment**): 

```Betti numbers 
sage get_db_homology.py "path/to/previous/output" "save_file_name" 
``` 

The saved file contains: 

* "bettis" of shape (n_architectures, n_trials, 5) recording the *i*th Betti number for i=0 to 4. 
* "archs" recording the architectures which are indexed by the n_architectures 

## Plotting Examples and Theorem 15

Samples of the plotting capability of this code are available in ```Example_Models.ipynb```, together
with the models given as an example in Theorem 15.

![image](https://user-images.githubusercontent.com/38443979/169736504-3299f4cc-07f0-4e81-846e-ac44817d984f.png)

