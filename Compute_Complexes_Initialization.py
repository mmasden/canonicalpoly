#Run this file as 
#python3 Initial_DB_Compute.py input_dimension, hidden_layers, minwidth, maxwidth, step, n_trials 
#For example 
#python3 Initial_DB_Compute.py 2 2 5 25 5 1



import torch 
from torch import nn 

import numpy as np 

from timeit import default_timer as timer

from polyhedra import cx 

import sys

### Define classes and functions 

class NeuralNetwork(nn.Module):
    def __init__(self, architecture):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        
        self.architecture=architecture

        self.linear_0 = nn.Linear(architecture[0],architecture[1])
        self.relu_0 = nn.ReLU()
        self.linear_1 = nn.Linear(architecture[1],architecture[2])
        if len(architecture)>3: 
            self.relu_1 = nn.ReLU()
            self.linear_2 = nn.Linear(architecture[2],architecture[3])
            if len(architecture)>4: 
                self.relu_2 = nn.ReLU()
                self.linear_3 = nn.Linear(architecture[3],architecture[4])
                
                if len(architecture)>5: 
                    print("Architecture too large")


    def forward(self, x):
        x = self.flatten(x)
        
        outs = []
        self.activity_0 = self.linear_0(x) 
        outs.append(self.activity_0)
        
        self.activity_1 = self.linear_1(self.relu_0(self.activity_0))
        outs.append(self.activity_1)
        
        if len(architecture)>3: 
            self.activity_2 = self.linear_2(self.relu_1(self.activity_1))
            outs.append(self.activity_2) 
            
            if len(architecture)>4: 
                self.activity_3 = self.linear_3(self.relu_2(self.activity_2))
                outs.append(self.activity_3)

        return outs

def TensorTupleToNumpy(tensortuple): 
    mylist = [int(thing.cpu().detach().numpy()) for thing in tensortuple]
    return tuple(mylist)


if __name__ == '__main__': 

    ### get trial parameters 
    input_dimension, hidden_layers, minwidth, maxwidth, step, n_trials = sys.argv[1:]

    ### File name and other parameters that can be set manually
    fname = "Initial_DBs_in{}_h{}_w{}_W{}_s{}_n{}".format(*sys.argv[1:])
    device = 'cpu'

    ### Actual script starts here 
    all_complexes = [] 
    all_points = [] 
    archs = []
    times = []

    ### 
    for n in range(int(minwidth),int(maxwidth), int(step)): 

        time0=timer() 
        
        architecture = [int(input_dimension)]+[int(n)]*int(hidden_layers)+[1]
        archs.append(architecture)

        all_complexes.append([])
        all_points.append([])
    
        for test in range(int(n_trials)): 
            
            model = NeuralNetwork(architecture).to(device)
            plotdict, points, ssv = cx.get_full_complex(model, max_depth=len(architecture)-1, device=device) 
            vertex_array = ssv.cpu().detach().numpy()
            points = points.cpu().detach().numpy() 
            
            all_complexes[-1].append(vertex_array) 
            all_points[-1].append(points) 
            
            print('.'*(test+1),end='\r')
        
        times.append(timer()-time0)

        print("\n architecture {} took time {} seconds".format(architecture,times[-1]) )
        
        #save at the end of each set of loops 
        np.savez("data/initialization/"+fname, 
                 complexes = np.array(all_complexes,dtype='object'), 
                 points = np.array(all_points, dtype='object'),
                 times = np.array(times),
                 archs = archs)     