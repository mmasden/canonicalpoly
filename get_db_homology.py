import sys
import numpy as np
from polyhedra import topology
import os

def betti_dict_to_list(dictionary, max_length): 
    templist=[0]*max_length
    for i in range(max_length):
        if i in dictionary.keys():
            templist[i]=int(dictionary[i])
    return templist
    

file = sys.argv[1]

savefile = sys.argv[2]


print("Opening {} to obtain decision boundaries.".format(file))
print("Saving at bettis_{}.npz.".format(savefile))

f = np.load(file,allow_pickle=True)

complexes = f["complexes"]
architectures = f["archs"]

all_bettis = []

#loop through each architecture in the file, and the corresponding polyhedral complexes

for i, (arch, cxs) in enumerate(zip(architectures,complexes)):
    all_bettis.append([])
    
    #loop through complexes
    
    for c in cxs: 
        
        #get the homology of the compactified decision boundary 
        bettis = topology.get_db_homology(c,arch)
        
        #it is returned as a dictionary, turn into a list
        bettilist = betti_dict_to_list(bettis,max_length=5)
        
        all_bettis[-1].append(bettilist)


    #save after every architecture 
    
    np.savez("bettis_{}".format(savefile), 
             bettis = np.array(all_bettis,dtype='object'), 
             archs = architectures,
             allow_pickle=True)