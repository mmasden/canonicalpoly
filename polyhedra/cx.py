import os
import torch
from torch import nn 
from torch.utils.data import DataLoader

import itertools 
import matplotlib.pyplot as plt
import numpy as np


class NeuralNetwork(nn.Module):
    def __init__(self, architecture):
        super(NeuralNetwork, self).__init__()
        
        self.architecture=architecture
        
        self.flatten = nn.Flatten()
        self.linear_0 = nn.Linear(architecture[0],architecture[1])
        self.relu_0 = nn.ReLU()
        self.linear_1 = nn.Linear(architecture[1],architecture[2])


    def forward(self, x):
        x = self.flatten(x)
        
        self.activity_0 = self.linear_0(x) 
        self.activity_1 = self.linear_1(self.relu_0(self.activity_0))

        return self.activity_0, self.activity_1
    

class DeepNeuralNetwork(nn.Module):
    def __init__(self, architecture):
        super(DeepNeuralNetwork, self).__init__()
        
        self.architecture=architecture
        
        self.flatten = nn.Flatten()
        self.linear_0 = nn.Linear(architecture[0],architecture[1])
        self.relu_0 = nn.ReLU()
        self.linear_1 = nn.Linear(architecture[1],architecture[2])
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(architecture[2],architecture[3])
        # self.relu_2 = nn.ReLU()
        # self.linear_3 = nn.Linear(architecture[3],architecture[4])


    def forward(self, x):
        x = self.flatten(x)
        
        self.activity_0 = self.linear_0(x) 
        self.activity_1 = self.linear_1(self.relu_0(self.activity_0))
        self.activity_2 = self.linear_2(self.relu_1(self.activity_1))
        #self.activity_3 = self.linear_3(self.relu_2(self.activity_2))

        return self.activity_0, self.activity_1, self.activity_2 #, self.activity_3



#multiply two sign sequences
def multiply(v1, v2): 
    '''Computes the product of two sign sequences''' 

    if len(v1)!= len(v2): 
        raise ValueError("The two sign sequences must have same length")
        return
    else: 
                     
        # apply product definition 

        product = list(v1) 
        for i in range(len(product)): 
            if product[i]==0: 
                product[i] = v2[i]
       
        return tuple(product)
    
def multiply_torch(v1, v2):
    '''Computes the product of two sign sequences given as Torch tensors''' 

    if not torch.is_tensor(v1): 
        v1 = torch.tensor(v1) 
    if not torch.is_tensor(v2): 
        v2=torch.tensor(v2)

    product = v1.clone()
    locs = torch.where(product==0)[0]
    product[locs]=v2[locs]
    
    return product
    
def edge_connected(v1, v2,  dim = 2): 
    '''Checks if two vertices, in given ambient dimension, are connected by an edge.
    This occurs if the product commutes and the resulting cell is an edge'''
     
    p1 = multiply(v1,v2)
    p2 = multiply(v2,v1) 
    #print(p1,p2)
    if p1==p2 and sum([p==0 for p in p1]) == dim-1: 
        return True 
    else: 
        return False 

def edge_connected_torch(v1,v2,dim=2):     
    '''Checks if two vertices, in given ambient dimension, are connected by an edge.
    This occurs if the product commutes and the resulting cell is an edge'''

    p1 = multiply_torch(v1,v2)
    p2 = multiply_torch(v2,v1) 
    #print(p1,p2)
    if torch.equal(p1,p2) and torch.sum(p1==0) == dim-1: 
        return True 
    else: 
        return False 
    
    
def is_face(v, F): 
    '''Checks if the cell represented by v is a face of the cell represented by F'''
    p = multiply(v,F)

    if p == tuple(F): 
        return True 
    else: 
        return False 
    
def is_face_torch(v, F): 
    '''Checks if the cell represented by v is a face of the cell represented by F'''

    p = multiply_torch(v,F)

    if torch.equal(p,F): 
        return True 
    else: 
        return False   
    
def make_affine(matrix, bias, device='cpu'):
    A = torch.hstack([matrix, bias.reshape(len(matrix),1)])
    A = torch.vstack([A,torch.zeros(1,A.shape[1],device=device)])
    A[-1,-1] = 1 
    return A 


def make_linear(affine_matrix): 
    matrix = affine_matrix[0:-1,0:-1]
    bias = affine_matrix[:-1,-1]
    return matrix,bias


def plot_complex(plot_dict, num_comparison, dim, ax=None, colors=None):
    ''' Plots the polyhedral complex with plot_dict in the form {ss:coordinates}'''

    if ax is None: 
        fix,ax =  plt.subplots(figsize=(5,5)) 
        ax.set_xlim((-10,10))
        ax.set_ylim((-10,10))

    
    if colors is None: 
        colors = ['black']*len(plot_dict)
         
    #for each pair of vertices: 
    for v in plot_dict: 
        for w in plot_dict: 

            #determine if they are connected by an edge
            if edge_connected(v[0:num_comparison],w[0:num_comparison], dim=dim): 
                
                hyper_set = set(np.where(np.array(v)==0)[0]).intersection(set(np.where(np.array(w)==0)[0]))
                hyper = max(hyper_set)
                
                #color the edge with the color of the latest hyperplane participating in the edge. 
                color = colors[hyper]
                
                if color=="black" or color=="blue":
                    ax.plot(*np.vstack([plot_dict[v],plot_dict[w]]).T, c=color,alpha=.1,zorder=1)
                elif color=="white": 
                    pass
                else:
                    ax.plot(*np.vstack([plot_dict[v],plot_dict[w]]).T, c=color,alpha=.5,zorder=1)
                    
    return ax
    

    
#obtain affine maps for each region 
def get_layer_map_on_region(ss,weights,biases,device='cpu'): 
    ''' 
    Inputs sign sequence in layer and parameter list for layer. Returns map on region of that layer.
    '''
    
    base_A = make_affine(weights,biases, device=device)
    region_indices = torch.where(ss==-1)[0]
    #print(region_indices)
    r_map = torch.clone(base_A)
    r_map[region_indices,:] = 0

    return r_map 

def tensor_tuple_to_numpy(tt): 
    
    tt = np.array([t.detach().numpy() for t in tt])
    
    return tuple(tt)

def numpyize_plot_dict(pd): 
    
    pd2 = {tensor_tuple_to_numpy(tt):pd[tt].detach().numpy() for tt in pd}
    
    return pd2
    




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
    #record the existing vertex ss as np arrays or similar
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
    
    image = model(points)

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
        if region_ss is None or is_face_torch(temp_ss[0:region_len],region_ss): 
            true_points.append(temp_pt) #.cpu().detach().numpy()) 
            true_ssv.append(temp_ss) 
    
    if len(true_ssv)>0: 
        true_ssv=torch.vstack(true_ssv)
        true_points=torch.vstack(true_points)
    
    
    return true_points, true_ssv
    
    
def get_all_maps_on_region(ss, depth, param_list, architecture, device='cpu'): 
       
    cumulative_architecture = [np.sum(architecture[1:i],dtype='int') for i in range(1,len(architecture)+1)]
    
    region_maps = []
    
    for i in range(depth): 
        layer_ss = ss[cumulative_architecture[i]:cumulative_architecture[i+1]]
        region_map_on_layer = get_layer_map_on_region(layer_ss,param_list[2*i],param_list[2*i+1], device=device)
        region_maps.append(region_map_on_layer)
        
        
    early_layer_maps = [region_maps[0]] 
    
    
    for rmap in region_maps[1:]:
        early_layer_maps.append(rmap @ early_layer_maps[-1])
    
        
    affine_layer_maps = [make_affine(param_list[2*i],param_list[2*i+1], device=device) for i in range(depth+1)]
    
    actual_layer_maps = [affine_layer_maps[0].detach()]
    
    for rmap, amap in zip(early_layer_maps, affine_layer_maps[1:]): 
        actual_layer_maps.append((amap@rmap).detach())
        
    return actual_layer_maps


def get_face_combos(ssr, existing_vertices): 
    '''Obtains minimal sets of hyperplanes forming the faces of a polyhedral region, 
       given a list of the sign sequences of its vertices '''

    combos=[]
    true_vertex_ssvs = [] 
    
    in_dim = sum(existing_vertices[0]==0)
    
    #get list of vertices which are a face of region with given ssr 
    
    for vertex in existing_vertices: 
        
        #truncate vertex sign sequence to appropriate length for layer
        
        if is_face_torch(vertex[0:len(ssr)],ssr): 
            true_vertex_ssvs.append(vertex[0:len(ssr)])
         
    true_vertex_ssvs = torch.vstack(true_vertex_ssvs)
    
    # all faces of C are the vertices of C with a number of their zeros
    # replaced by the corresponding entry of C 
    
    # the bent hyperplanes which intersect to form the faces of C are given 
    # by all subsets of the hyperplanes which intersect to form the vertices of C 
    # e.g. if BH 1n2n3 is an vertex of C, then 1n2 intersect to form a face, 2n3, 1n3 and
    # 1, 2, and 3 individually 
    
    # e.g. simplicial closure 
    
    # This should be shellable, so there should be a way to do this
    # which is much faster than below. 
    
    top_simplices = [torch.where(vertex==0)[0] for vertex in true_vertex_ssvs]
    top_simplices = torch.vstack(top_simplices)
    
    hyperplane_combos = [[]] 

    #loop through size of hyperplane combo 
    
    for i in range(1, in_dim+1): 

        #get all i-subsets of top simplices 

        hyperplane_combos.append([])
        
        combinations = torch.combinations(torch.arange(in_dim),r=i) 
        
        
        temp_h_combos = torch.vstack([top_simplices[:,combination] for combination in combinations])
        
        temp_h_combos = torch.unique(temp_h_combos,dim=0) 
        
        hyperplane_combos[i]=temp_h_combos                                    
        
    return hyperplane_combos


#find possible intersections of bent hyperplanes, given a list of all neurons in earlier layers 
# and a list of neurons in later layers.

def find_intersections(in_dim, last_layer, last_biases, image_dim, ssr, architecture,hyperplane_combos=None, early_layer_maps=None, early_layer_biases=None, device='cpu'): 
    '''Given a polyhedral region R, in input space, layer_maps is a tensor of the activity functions
    of each neuron on the interior of that region. If this is the first layer, input None. 
    last_layer is the single layer after layer_maps which provides "new" bent hyperplanes.
    Returns the locations of the vertices, and which pairs of bent hyperplanes intersect 
    at those points.
    
    Returns: locations of points which represent possible vertices, the pairs of hyperplanes which 
    intersect to make those points'''
        
    last_layer = last_layer.detach()
    last_biases = last_biases.detach() 
    
    #If the last layer is the only layer then layer_maps is None. 
    if early_layer_maps is None or early_layer_biases is None: 
        
        #get at tensorfied list of all neurons in the output of the given layer 
        n_out=torch.arange(len(last_layer)) 
        
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
        
        # loop through k, the number of new bent hyperplanes involved in intersection 
        # The number of new bent hyperplanes involved in the intersection 
        # is bounded above by the dimension of the image of the region in this layer! 
        
        # omit 0 because must include some from new layers; can't go above n_out
        for k in range(1,min(image_dim+1,in_dim,len(last_biases)+1)):  
            
            last_combos = torch.combinations(n_out, r=k) 
                      
            early_combos = hyperplane_combos[in_dim-k]
            
            old_vals=len(early_layer_maps)
            
            # worry about degeneracy only if it has been collapsed 
            
            if image_dim < in_dim:
                
                if image_dim ==0: 
                    pass
                else: 
                    
                    
                    # IF HYPERPLANES NONGENERIC SKIP 
                    # This occurs if image_dim < in_dim (the region has been collapsed) 
                    # and the bent hyperplanes from earlier layers intersect in a region 
                    # sent to too few dimensions to generically intersect with the. 
                    # next layer's hyperplanes.
                    # The latter occurs when, if taking the sign sequence of the region
                    # and setting all the BH's coordinates to 0, you have fewer 1's left than 
                    # the dimension minus the number of new hyperplanes (in_dim - k)
                    # that is, the rank is too low 
                    
                    remaining_dims = ssr.repeat((len(early_combos),1))
                    
                    for i in range(len(remaining_dims)):
                        remaining_dims[i, early_combos[i]]=-1 
                    
                    
                    total_ones = tensor_region_image_dimension(remaining_dims, architecture, device=device)
                    
                    good_initial_BHs = total_ones >= in_dim - k
                    
                    good_early_combos = early_combos[good_initial_BHs]

                    temporary_maps = torch.vstack([early_layer_maps, last_layer])
                    temporary_biases = torch.vstack([torch.reshape(early_layer_biases, [-1,1]), torch.reshape(last_biases,[-1,1])])

                    total_combos = torch.hstack([good_early_combos.repeat((len(last_combos),1)), last_combos.repeat_interleave(len(good_early_combos),dim=0)+old_vals])
                    points = torch.linalg.solve(temporary_maps[total_combos], -temporary_biases[total_combos])
                    
                    all_points.append(points.reshape([-1,in_dim]))
                    combos.extend(total_combos)
                
            else: 

                #turn early_layer_maps and last_layer into one stack 
                
                temporary_maps = torch.vstack([early_layer_maps, last_layer])

                temporary_biases = torch.vstack([torch.reshape(early_layer_biases, [-1,1]), torch.reshape(last_biases,[-1,1])])
                
                total_combos = torch.hstack([early_combos.repeat((len(last_combos),1)), last_combos.repeat_interleave(len(early_combos),dim=0)+old_vals])

                points = torch.linalg.solve(temporary_maps[total_combos], -temporary_biases[total_combos])
                
                all_points.append(points.reshape([-1,in_dim]))
                combos.extend(total_combos)
                
                
        
        if len(all_points)>0: 
            all_points=torch.vstack(all_points)
        
        #add in intersections of only the last layer 
        #UNLESS the most recent output is singular. 
        
        
        if len(last_biases)<in_dim or image_dim < in_dim:
            pass
        
        else:
            last_combos = torch.combinations(n_out, r=in_dim)
            
            temp_points = torch.linalg.solve(last_layer[last_combos],-last_biases[last_combos])
            
            #print(all_points, temp_points)
            
            all_points = torch.vstack([all_points,temp_points])
            
            last_combos = list((last_combos+old_vals))

            combos.extend(last_combos)
        
        
        return all_points, combos
        
                
def region_image_dimension(temp_ssr, architecture, depth=None): 
    
    # all top-dim regions begin at n_0-dimensional
    current_dim = architecture[0]
    
    # need to get sign sequences corresponding to individual layers
    cumulative_widths = [0]+[sum(architecture[1:i]) for i in range(2,len(architecture))]
    
    #loop through layer widths until depth
    for i,layerwidth in enumerate(architecture[1:depth+1]): 
        
        #get sign sequence corresponding with most recent layer 
        layer_neurons = temp_ssr[cumulative_widths[i]:cumulative_widths[i+1]]
        
        # the number of 1's in the sign sequence is the 
        # maximum dimension of the image of the region 
        
        max_dim = sum([s == 1 for s in layer_neurons])
                                  
        # generically the dimension will either stay the same 
        # or be collapsed to the dimension of the image of the map
        
        # eg a 1d subspace of R^2 is sent to all of R under a linear map
        # R^2->R unless it is in the kernel of the map which is a 
        # nongeneric condition
         
        current_dim = min(current_dim,max_dim)
    
    return int(current_dim)


def tensor_region_image_dimension(tensor_ssr, architecture, device): 
    
    # all top-dim regions begin at n_0-dimensional
    current_dim = torch.tensor([architecture[0]],device=device).repeat(len(tensor_ssr))
    
    # need to get sign sequences corresponding to individual layers
    cumulative_widths = [0]+[sum(architecture[1:i]) for i in range(2,len(architecture))]
    
    #find the depth to stop at 
    
    depth = torch.where(torch.Tensor(cumulative_widths)==len(tensor_ssr[0]))[0][0].numpy()
    #print(depth)
    
    #loop through layer widths until depth
    for i,layerwidth in enumerate(architecture[1:depth+1]): 
        
        #get sign sequence corresponding with most recent layer 
        layer_neurons = tensor_ssr[:,cumulative_widths[i]:cumulative_widths[i+1]]
        
        # the number of 1's in the sign sequence is the 
        # maximum dimension of the image of the region 
        
        max_dim = torch.sum(layer_neurons==1, axis=1)
                                  
        # generically the dimension will either stay the same 
        # or be collapsed to the dimension of the image of the map
        
        # eg a 1d subspace of R^2 is sent to all of R under a linear map
        # R^2->R unless it is in the kernel of the map which is a 
        # nongeneric condition
        current_dim = torch.min(torch.vstack([current_dim,max_dim]),axis=0).values
    
    return current_dim

        
def get_full_complex(model, max_depth=None, device=None, mode='solve', verbose=False): 
    '''assumes model is feedforward and has appropriate structure.
    Outputs dictionary with vertices' signs and locations of vertices.''' 
    
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
    
    signs = torch.tensor(get_signs(in_dim), device=device).float()

    
    #get first layer sign sequences.
    
    if mode == 'solve':
        temp_points, temp_combos = find_intersections(in_dim,parameters[0],parameters[1], None, None, architecture, device=device)
    else: 
        print("Mode invalid.")
        
    #initialize full list of points, sign sequences, and ss_dict  
    all_points, all_ssv = determine_existing_points(temp_points,temp_combos,model, device=device)
    
    if verbose: 
        print("First Layer Complete")
    
    tsv=all_ssv.clone() #.cpu().detach().numpy()
    
    all_ss_dict = {tuple(ss.int()): pt for ss, pt in zip(tsv,all_points)}

    # get subsequent layer sign sequences 
    # requires updating points, ssv and ss_dict 
    
    #loop through layers 
    for i in range(1, len(architecture)-1):

        #obtain regions which are present from previous layer 
        
        ssr = get_ssr(all_ssv,sum(architecture[1:i+1]), signs)
        
        #initialize placeholder for new points and ssv 
        new_points, new_ssv = [],[]
        
        #loop through regions from previous layers
        
        num_ssr = len(ssr)
        
        if verbose: 
            print("{} regions to evaluate ... ".format(num_ssr))
            
        for counter,temp_ssr in enumerate(ssr):
            
            # obtain the maps on the region induced by the model at each depth
            # note i = layer depth 
            # parameters = list of model parameters
            
            region_maps = get_all_maps_on_region(temp_ssr,i,parameters,architecture, device=device)
            
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
            
            #get the dimension of the image of the region in this layer
            
            image_dim = region_image_dimension(temp_ssr, architecture, depth=i)
                       
            #get list of faces of region! 
            
            hyperplane_combos = get_face_combos(temp_ssr, all_ssv)
            
            #get temporary list of points 
            
            if mode == 'solve':
                temp_points,temp_combos = find_intersections(in_dim, 
                                   last_layer, 
                                   last_biases, 
                                   image_dim, temp_ssr, architecture,
                                   hyperplane_combos = hyperplane_combos,
                                   early_layer_maps=early_layer_maps, 
                                   early_layer_biases=early_layer_biases,
                                   device=device)

            else: 
                print("Mode invalid")
                return
            
            #if there's at least one point evaluate whether they belong to C(F)

            if len(temp_points)>0: 
                temp_pts, temp_ssv = determine_existing_points(temp_points,
                                                                temp_combos, 
                                                               model, 
                                                               region_ss=temp_ssr, device=device)

                new_points.extend(temp_pts)
                new_ssv.extend(temp_ssv)
                
            if verbose:
                print("*"*int(counter/num_ssr*20+1), 
                      "."*(20-int(counter/num_ssr*20)-1),
                      " {percent:.2f}%".format(percent=counter/num_ssr*100),
                      end='\r')
                        
        #done looping through regions, now collect points 
       
        if verbose: 
            print("\n Layer {} complete.".format(i+1))
            
        if len(new_points)>0: 

            new_points=torch.vstack(new_points)

            all_points = torch.vstack([all_points,new_points])
            
            new_ssv = torch.vstack(new_ssv)
            all_ssv = torch.vstack([all_ssv,new_ssv]) 
    
            new_ss_dict = {tuple(ss):pt for ss,pt in zip(new_ssv,new_points)}
            
            all_ss_dict = all_ss_dict | new_ss_dict
        
        else:
            pass
        
    
    return all_ss_dict, all_points, all_ssv
        
def make_sphere(dim,n): 
    
    temppts = torch.normal(0.0,1.0,(n,dim))  
    points0 = 2*temppts/torch.linalg.norm(temppts,dim=1).reshape(n,1)  
    scatter = 0.05*torch.normal(0,1,(n,1)) 
    points0=points0*scatter+points0
    
    points1 = 0.2*torch.normal(0.0,1.0, (n,dim))
    
    points=torch.vstack([points0,points1])
    labels=torch.hstack([torch.zeros(n),torch.ones(n)]).reshape(2*n,1)
    
    return points,labels

def make_torus(n): 
        
    thetas = 2*torch.pi*torch.rand(n)
    phis = 2*torch.pi*torch.rand(n)
    
    xs = 4*torch.cos(thetas)-2*torch.cos(thetas)*torch.cos(phis)
    ys = 4*torch.sin(thetas)-2*torch.sin(thetas)*torch.cos(phis)
    zs = 2*torch.sin(phis)
    
    pts0 = torch.vstack([xs,ys,zs]).T
    
    pts1,_ = make_sphere(2,n)
    
    pts1 = 2*torch.hstack([pts1[0:n],torch.zeros((n,1))])
    
    
    points = torch.vstack([pts0,pts1])
    labels=torch.hstack([torch.zeros(n),torch.ones(n)]).reshape(2*n,1)
    
    return points,labels