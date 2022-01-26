import os
import torch
from torch import nn 
from torch.utils.data import DataLoader
#from torchvision import datasets, transforms 

import sklearn
import itertools 
import matplotlib.pyplot as plt
import numpy as np

#multiply two sign sequences
def multiply(v1, v2): 
    if len(v1)!= len(v2): 
        raise ValueError("The two sign sequences must have same length")
        return
    else: 
        product = list(v1) 
        for i in range(len(product)): 
            if product[i]==0: 
                product[i] = v2[i]
        
        return tuple(product)
    
#check if two vertices, in ambient dimension dim, are edge connected 
def edge_connected(v1, v2,  dim = 2): 
    p1 = multiply(v1,v2)
    p2 = multiply(v2,v1) 
    #print(p1,p2)
    if p1 == p2 and np.sum([p == 0 for p in p1 ]) == dim-1: 
        return True 
    else: 
        return False 

# checks if v is a face of F
def is_face(v, F): 
    p = multiply(v,F)

    if p == tuple(F): 
        return True 
    else: 
        return False 
    
# Linear algebra utilities #
def make_affine(matrix, bias, device='cpu'):
    A = torch.hstack([matrix, bias.reshape(len(matrix),1)])
    A = torch.vstack([A,torch.zeros(1,A.shape[1]).to(device)])
    A[-1,-1]=1 
    return A 


def make_linear(affine_matrix): 
    matrix = affine_matrix[0:-1,0:-1]
    bias = affine_matrix[:-1,-1]
    return matrix,bias


        
def plot_complex(plot_dict, num_comparison, ax=None, colors=None):
    if ax is None: 
        fix,ax =  plt.subplots(figsize=(5,5)) 
        ax.set_xlim((-10,10))
        ax.set_ylim((-10,10))
        
    else:
        pass 
    
    if colors is None: 
        colors = ['black']*len(plot_dict)
         
    
    for v in plot_dict: 
        for w in plot_dict: 
            if edge_connected(v[0:num_comparison],w[0:num_comparison]): 
                
                hyper_set = set(np.where(np.array(v)==0)[0]).intersection(set(np.where(np.array(w)==0)[0]))
                hyper = hyper_set.pop()
                color = colors[hyper]
                
                if color=="black" or color=="blue":
                    ax.plot(*np.vstack([plot_dict[v],plot_dict[w]]).T, c=color,alpha=.1,zorder=1)
                else:
                    ax.plot(*np.vstack([plot_dict[v],plot_dict[w]]).T, c=color,alpha=.5,zorder=1)
                    
    return ax
    
    
    
    
#obtain affine maps for each region 
def get_layer_map_on_region(ss,weights,biases,device='cpu'): 
    ''' 
    Inputs sign sequence IN LAYER and parameter list FOR LAYER. Returns map on region OF THAT LAYER
    '''
    
    base_A = make_affine(weights,biases, device=device)
    region_indices = torch.where(ss==-1)[0]
    #print(region_indices)
    r_map = torch.clone(base_A)
    r_map[region_indices,:] = 0

    return r_map 

    
## NN Setup ###

# class NeuralNetwork(nn.Module):
#     def __init__(self, architecture):
#         super(NeuralNetwork, self).__init__()
#         self.flatten = nn.Flatten()
        
#         self.linears = []
#         self.relus = []
        
#         for i in range(len(architecture)-2):
#             self.linears.append(nn.Linear(architecture[i],architecture[i+1]))
#             self.relus.append(nn.ReLU())
        
#         self.linears.append(nn.Linear(architecture[-2],architecture[-1]))
#         print(self.linears)

#     def forward(self, x):
#         x = self.flatten(x)
        
#         self.activities = []
                       
#         self.activities.append(self.linears[0](x)) 
        
#         for i in range(len(architecture)):               
#             self.activities.append(self.linears[i+1](self.relus[i](self.activities[i])))

#         return self.activities


class NeuralNetwork(nn.Module):
    def __init__(self, architecture):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()

        self.linear_0 = nn.Linear(architecture[0],architecture[1])
        self.relu_0 = nn.ReLU()
        self.linear_1 = nn.Linear(architecture[1],architecture[2])
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(architecture[2],architecture[3])
        self.relu_2 = nn.ReLU()
        self.linear_3 = nn.Linear(architecture[3],architecture[4])


    def forward(self, x):
        x = self.flatten(x)
        
        self.activity_0 = self.linear_0(x) 
        self.activity_1 = self.linear_1(self.relu_0(self.activity_0))
        self.activity_2 = self.linear_2(self.relu_1(self.activity_1))
        self.activity_3 = self.linear_3(self.relu_2(self.activity_2))

        return self.activity_0, self.activity_1, self.activity_2, self.activity_3


def get_signs(dim):
     #[[1,1],[-1,-1],[1,-1],[-1,1]]      
    if dim == 1: 
        return [[1],[-1]]
    elif dim > 1: 
        signs = [] 
        for signlist in get_signs(dim-1):
            signs.append(signlist+[1]) 
            signs.append(signlist+[-1]) 
        
        return signs
    
    elif dim <1: 
        print("not valid")
        return


    
def get_ssr(ssv, ss_length, signs): 
    #record the existing vertex ss as np arrays
    ssv_np = [ss[0:ss_length] for ss in ssv]
    
    #record the regions that are present as a set 
    ssr = []

    #loop through vertices and obtain regions which are adjacent
    for ss in ssv_np: 
        #ss_np = np.array(ss)
        locs = torch.where(ss==0)[0]
        dimension=len(locs)
        
        tempss=torch.clone(ss)
        #print(tempss)
        #print(ss)

        for sign in signs: 
            #print(sign)
            #print(tempss[locs])
            tempss[locs]=sign
            #print(tempss)
            
            ssr.append(tempss.clone())
                
                #print(ssr)
    
    ssr=torch.vstack(ssr)
    
    return torch.unique(ssr, dim=0)
    
    
def determine_existing_points(points, combos, model, region_ss=None, device='cpu'):
    ''' evaluates sign sequence of points matches existing sign sequence in region
    Region sign sequence should be truncated.'''
    
    image = model(points.to(device))

    # obtains sign sequence of initial vertices
    ssv = torch.hstack([torch.sign(image[i]) for i in range(len(image))])

    #force correct signs at intersections 
    for i in range(len(ssv)): 
        ssv[i,combos[i]]=0

    true_points = []
    true_ssv = []

    #determine if it is a face 
    
    region_len = 0 if region_ss is None else len(region_ss) 
    
    for  temp_pt, temp_ss in zip(points, ssv): 
        #temp_ss = temp_ss.cpu().detach().numpy() try not detaching
        if region_ss is None or is_face(temp_ss[0:region_len],region_ss): 
            true_points.append(temp_pt) #.cpu().detach().numpy()) 
            true_ssv.append(temp_ss) 
    
    if len(true_ssv)>0: 
        true_ssv=torch.vstack(true_ssv)
        true_points=torch.vstack(true_points)
            #true_points=np.array(true_points.cpu().detach().numpy())
    
    return true_points, true_ssv
    
    
def get_all_maps_on_region(ss, depth, param_list, architecture, device='cpu'):    
    cumulative_architecture = [np.sum(architecture[1:i],dtype='int') for i in range(1,len(architecture)+1)]
    #print(cumulative_architecture)
    
    region_maps = []
    
    for i in range(depth): 
        layer_ss = ss[cumulative_architecture[i]:cumulative_architecture[i+1]]
        region_map_on_layer = get_layer_map_on_region(layer_ss,param_list[2*i],param_list[2*i+1], device=device)
        #print(region_map_on_layer)
        region_maps.append(region_map_on_layer)
        
        
    early_layer_maps = [region_maps[0]] 
    
    #need: all earlier maps with zeroed spots 
    
    for rmap in region_maps[1:]:
        early_layer_maps.append(rmap @ early_layer_maps[-1])
    
    #print(early_layer_maps)
    
    #last map not zeroed for each neuron 
    
    affine_layer_maps = [make_affine(param_list[2*i],param_list[2*i+1], device=device) for i in range(depth+1)]
   # print(affine_layer_maps)
    
    actual_layer_maps = [affine_layer_maps[0].detach()]
    
    for rmap, amap in zip(early_layer_maps, affine_layer_maps[1:]): 
        actual_layer_maps.append((amap@rmap).detach())
        
    return actual_layer_maps



#find POSSIBLE intersections of bent hyperplanes, given a list of all neurons in earlier layers 
# and a list of neurons in later layers. 
def find_intersections(in_dim, last_layer, last_biases, early_layer_maps=None, early_layer_biases=None, device=None): 
    '''Given a polyhedral region R, in input space, layer_maps is a tensor of the activity functions
    of each neuron on the interior of that region. If this is the first layer, input None. 
    last_layer is the single layer after layer_maps which provides "new" bent hyperplanes.
    Returns the locations of the vertices, and which pairs of bent hyperplanes intersect 
    at those points.
    
    Returns: locations of points which represent possible vertices, the pairs of hyperplanes which 
    intersect to make those points'''
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    last_layer = last_layer.detach()
    last_biases = last_biases.detach() 
    
    #If the last layer is the only layer then layer_maps is None. 
    if early_layer_maps is None or early_layer_biases is None: 
        #get at tensorfied list of all neurons in the output of the given layer 
        n_out=torch.tensor(range(len(last_layer))).to(device)
        
        #obtain all in_dim-combinations of the first layer's hyperplanes  
        combos = torch.combinations( n_out, r=in_dim)
            
        #solves for points
        points = torch.linalg.solve(last_layer[combos].detach(), -last_biases[combos].detach()) 

        return points, combos
    
    else:
        all_points = []
        combos = []

        n_between = torch.tensor(range(len(early_layer_biases)))
        n_out = torch.tensor(range(len(last_biases)))
        
        
        for k in range(1,min(in_dim,len(last_biases)+1)):  #omit 0 because must include some from new layers; can't go above n_out
            last_combos = torch.combinations(n_out, r=k) 
            early_combos = torch.combinations(n_between, r = in_dim - k)
            
            old_vals=len(early_layer_maps)
            #print(k)
            
            #can this be done without a loop? 
            for i,j in torch.cartesian_prod(torch.tensor(range(len(early_combos))),
                                            torch.tensor(range(len(last_combos)))):
                early_neurons = early_combos[i]
                last_neurons = last_combos[j]
                
                #print(torch.vstack([early_layer_maps[early_neurons],last_layer[last_neurons]]))
                
                #if torch.det?? some pairs of hyperplanes might be parallel here.
                
                
                
                
                if torch.linalg.det(torch.vstack(
                                [early_layer_maps[early_neurons],
                                 last_layer[last_neurons]])) ==0: 
                    pass
                else: 

                    point = torch.linalg.solve(
                                torch.vstack(
                                    [early_layer_maps[early_neurons],
                                     last_layer[last_neurons]]), 
                                -torch.vstack(
                                    [torch.reshape(early_layer_biases[early_neurons],[-1,1]),
                                     torch.reshape(last_biases[last_neurons],[-1,1])]))

                    all_points.append(point.reshape((1,in_dim)))
                    combos.append(list(early_neurons.numpy())+list(last_neurons.numpy()+old_vals))
        
        if len(all_points)>0: 
            all_points=torch.vstack(all_points)
        
        #add in intersections of only the last layer 
        #UNLESS the most recent output is singular. 
        #in which case ... all will be singular?? 
        
        if len(last_biases)<in_dim:
            pass;
        elif torch.linalg.det(last_layer[0:in_dim])==0: 
            pass
        else:

            last_combos = torch.combinations(n_out, r=in_dim)

            temp_points = torch.linalg.solve(last_layer[last_combos],-last_biases[last_combos])
            #temp_points = list(temp_points.cpu().numpy())
            all_points = torch.vstack([all_points,temp_points])
            last_combos = list((last_combos+old_vals).numpy())
            #print(last_combos)
            combos.extend(last_combos)
        
        
        return all_points, np.array(combos)
        

        
        
def get_full_complex(model, max_depth=None, device=None): 
    '''assumes model is feedforward and has appropriate structure.
    Outputs dictionary with vertices' signs and locations of vertices
    This is a less efficient way than is possible, but it's the first 
    one I thought of''' 
    
    if device is None: 
        device='cpu'
    
    parameters = list(model.parameters())
    
    if max_depth is None: 
        depth = len(parameters)//2 
    else: 
        depth = max_depth
    
    architecture = [parameters[0].shape[1]] #input dimension 
    
    for i in range(depth): 
        architecture.append(parameters[2*i].shape[0]) #intermediate dimensions 
        
    architecture = tuple(architecture) 
    
    in_dim = architecture[0]
    
    signs = torch.Tensor(get_signs(in_dim)).to(device)

    
    #get first layer sign sequences.
    temp_points, temp_combos = find_intersections(in_dim,parameters[0],parameters[1])
    
    #initialize full list of points, sign sequences, and ss_dict  
    all_points, all_ssv = determine_existing_points(temp_points,temp_combos,model, device=device)
    
    tsv=all_ssv.clone().cpu().detach().numpy()
    #tpt = all_points.clone().cpu().detach().numpy()
    
    #all_ss_dict = {tuple(ss): pt for ss, pt in zip(tsv,tpt)}
    
    all_ss_dict = {tuple(ss): pt for ss, pt in zip(tsv,all_points)}
    #print(all_points)
    #print(all_points)

    # get subsequent layer sign sequences 
    # requires updating points, ssv and ss_dict 
    
    #loop through layers 
    for i in range(1, len(architecture)-1):
        #obtain regions which are present from previous layer 
        #print(sum(architecture[1:i+1]))
        ssr = get_ssr(all_ssv,sum(architecture[1:i+1]), signs)
        #print(len(ssr))
        #initialize placeholder for new points and ssv 
        new_points, new_ssv = [],[]
        #print(i)
        
        #loop through regions from previous layers 
        for temp_ssr in ssr:
            #obtain the maps on the region induced by the model 
            region_maps = get_all_maps_on_region(temp_ssr,i,parameters,architecture, device=device)
            #print(len(region_maps))
            #obtain the early layer maps as a list  of weights and biases
            early_layer_maps, early_layer_biases=[],[]
            for j in range(i):
                ll, bb = make_linear(region_maps[j])
                early_layer_maps.extend(ll) 
                early_layer_biases.extend(bb) 
            
            
            early_layer_maps=torch.vstack(early_layer_maps)
            early_layer_biases = torch.vstack(early_layer_biases)
            
            #obtain the last layer map as a list of weights and biases 
            last_layer, last_biases = make_linear(region_maps[-1])

            
            #get temporary list of points 
            temp_points,temp_combos = find_intersections(in_dim, last_layer, last_biases, 
                                   early_layer_maps=early_layer_maps, 
                                   early_layer_biases=early_layer_biases,
                                   device=device)
            
            #if there's at least one point evaluate the veracity of it
            if len(temp_points)>0: 
                temp_pts, temp_ssv = determine_existing_points(temp_points,
                                                                  temp_combos,model, region_ss=temp_ssr, device=device)

                new_points.extend(temp_pts)
                new_ssv.extend(temp_ssv)
        
        #done looping through regions, now collect points 
       
        if len(new_points)>0: 
            #print(new_points)
            new_points=torch.vstack(new_points)
            #print(all_points)
            all_points = torch.vstack([all_points,new_points])
            
            new_ssv = torch.vstack(new_ssv)
            all_ssv = torch.vstack([all_ssv,new_ssv]) 
            
            
            
            # tsv = new_ssv.clone().cpu().detach().numpy()
            # tpt = new_points.clone().cpu().detach().numpy()

            new_ssv=new_ssv.cpu().detach().numpy()
            # new_ssv = np.array(new_ssv,dtype='int')
            new_ss_dict = {tuple(ss):pt for ss,pt in zip(new_ssv,new_points)}
            
            all_ss_dict = all_ss_dict | new_ss_dict

        else:
            pass
    
    return all_ss_dict, all_points, all_ssv
        
        