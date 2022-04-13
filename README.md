# polytorch

This code computes the polyhedral complex of a ReLU Neural Network in Pytorch by computing only the vertices and their sign sequences. This allows for computation of topological invariants of subcomplexes of the polyhedral complex, for example, its decision boundary. 

Dependencies / currently developed with: 

* Python 3.9+ 
* matplotlib
* scikit-learn
* numpy

"cx" module:
* PyTorch 1.11

"topology" module: 
* Sage 9.0
