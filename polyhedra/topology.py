# This module requires Sage to function fully. 

from sage.all import Zmod, matrix
import numpy as np


def cell_coboundary(cell): 
    ''' Computes the set of cells which are the coboundary of a cell in the polyhedral complex.
        If s_{i} is a cell element equal to 0 then it can be replaced by +- 1 to find a coboundary element. 
    '''
    
    
    cobdy = set()
    
    for i in range(len(cell)): 
        entry = cell[i]
        if entry==0: 
            newcell = list(cell) 
            newcell[i]=-1 
            cobdy.add(tuple(newcell))
            newcell[i]=1 
            cobdy.add(tuple(newcell)) 
            
    
    return cobdy


def bh_coboundary(cell, bh=-1): 
    ''' Computes the set of cells in the coboundary of a cell, restricted to *within* a particular 
        bent hyperplane denoted by bh. Does this by forcing all entries in bh to 0 (not necessarily the fastest!)
        
        By default, computes the coboundary of a cell in the last bent hyperplane (the decision boundary).
        
        Passing a np array of integers as bh MIGHT work to slice the tuple so as to compute the coboundary 
        within an intersection of bent hyperplanes... fancy.
        
    '''
    
    cobdy = set()
    
    #only do coboundary within the intersection 
    
    for i in np.delete(range(len(cell)),bh): 
        entry = cell[i]
        
        if entry==0: 
            newcell = list(cell) 
            newcell[i]=-1 
            
            #hack to make sure that index of interest stays 0
            #newcell[bh]=0
            
            cobdy.add(tuple(newcell))
            newcell[i]=1 

            #hack to make sure that index of interest stays 0
            #newcell[bh]=0

            cobdy.add(tuple(newcell)) 
    
    return cobdy

def complex_coboundary(vertex_list, architecture, bh = -1):
    
    ''' Computes the cochain complex of the polyhedral complex associated with a particular
    bent hyperplane or slice of bent hyperplanes. 
    
    Returns a list of sets of faces in each dimension, together with a dictionary defining the
    map sending each face to its coboundary. 
    
    This does not include the "point at infinity." Some faces are unbounded.
    ''' 

    db_vertices = vertex_list[vertex_list[:,-1]==0]

    coboundary_map = dict() 

    faces = [[] for i in range(architecture[0])] #up to n-1 dimensional
    
    faces[0] = set([tuple(np.array(vtx,dtype='int')) for vtx in db_vertices])
    
    comaps = [ [] for i in range(1,architecture[0])]

    for i in range(architecture[0]-1): 
        
        facesip1 = set()
        
        for element in faces[i]: 
            coelts = bh_coboundary(element, bh = bh)
            coboundary_map[tuple(np.array(element,dtype="int"))] = coelts
            
            facesip1 = facesip1.union(coelts)
        
        faces[i+1] = facesip1      
            
    return faces, coboundary_map
        

def get_coboundary_matrices(faces, coboundary_map, architecture, field=Zmod(2)): 
    
    ''' 
     Computes the coboundary matrices of the cochain complex from the maps defined in 
     complex_coboundary
    '''
    
    d = []
    
    d.append(matrix(field, 0, len(faces[0])+1, sparse=True))

    flists = [list(f) for f in faces]
    fkeys = []
    
    for i in range(1, architecture[0]): 
        d.append( matrix(field, d[i-1].dimensions()[1], len(flists[i]), sparse=True) )
        
    for i in range(architecture[0]):
        fkeys.append({flists[i][j]:j for j in range(len(flists[i]))})
    
    #construct boundary matrices 

    #get each dim coboundary except the point at infinity
    #for each dimension 
    
    for i in range(architecture[0]-1):
        #loop through cells in that dimension  using j
        for j in range(len(flists[i])): 
            #go through coboundary elements of that cell and add them to d 
            #print(flists[i][j])
            for entry in coboundary_map[tuple(flists[i][j])]: 
                
                #obtain key of that cell as k
                k = fkeys[i+1][entry]
                
                #boundary shows up once for each cell due to convexity 
                d[i+1][j,k]=1

    # add point at infinity to ends of all unbounded edges so this actually works as a coboundary map
    for j in range(len(flists[1])): 
        boundary=sum(d[1][:,j])[0]
        if boundary==1: 
            d[1][-1,j]=1
            
    return d


