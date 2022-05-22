# Algorithmic Determination of the Combinatorial Structure of the Linear Regions of ReLU Neural Networks

This repository is the official implementation of _Algorithmic Determination of the Combinatorial Structure of the Linear Regions of ReLU Neural Networks_.

The included code computes the polyhedral complex of a ReLU Neural Network in Pytorch by computing only the vertices and their sign sequences. This allows for computation of topological invariants of subcomplexes of the polyhedral complex, for example, its decision boundary. 

![torus](https://user-images.githubusercontent.com/38443979/169712774-31db512e-1e8b-4e00-b8fc-02d6bf4d3d0f.png)

## Requirements

To install requirements for obtaining the polyhedral decomposition of input space,run the following in a Python 3.9+ virtual environment.

```setup
pip install -r requirements_polyhedra.txt
```

For obtaining the topological decomposition of input space, we use Sage 9.0, with installation instructions provided [here](https://doc.sagemath.org/html/en/installation/index.html). No additional requirements are necessary.

## Obtaining Polyhedral Complexes

To obtain the polyhedral complexes for random initializations of neural networks, run:

```polyhedral complex
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```
For example, the command ``` ```  

The saved file ``` asdf.npz``` contains two 

To obtain the Betti numbers of the resulting one-point compactified decision boundary, 

The saved file ``` . ``` contains . 



## Obtaining Topological Data

To obtain the topological properties of the 

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

## Pre-Generated Polyhedral Complexes

The data obtained in the experiments for this :

## Results


| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

