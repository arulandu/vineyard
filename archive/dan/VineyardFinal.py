# Matrix Operations
import numpy as np

# Plotting
from matplotlib import pyplot as plt
import plotly.graph_objects as go

# TDA and Optimization
import gudhi as gd
import gudhi.wasserstein
import gudhi.hera
import ot

import scipy
from scipy.optimize import linear_sum_assignment

# constants
INT_MAX = 2147483647

###############################################
## Persistence and Straight Line Homotopy ##

def persistence(array, fmax=255., inf=INT_MAX, dimension=None, invert=True):
    height, width = array.shape
    cubeComplex = gd.CubicalComplex(
        dimensions = [width,height],
        top_dimensional_cells = fmax - array.flatten() if invert else array.flatten()
    )
 
    if dimension == None:
        persistence = cubeComplex.persistence()
    else:
        cubeComplex.compute_persistence()
        persistence = cubeComplex.persistence_intervals_in_dimension(dimension)
        persistence[np.isinf(persistence)] = inf
        
    return persistence

def Get_Adjacency_Persistence(graph, column, dimension = None, popCol = 'TOTPOP', popMin = 10):
    scomplex = gd.SimplexTree()

    for v in graph.nodes:
        scomplex.insert([v])

    for e in graph.edges:
        scomplex.insert([e[0], e[1]])

    for v in graph.nodes:
        if graph.nodes[v][f'{popCol}'] >= popMin:
            scomplex.assign_filtration(
            [v], #we have to put [] here because a 0-simplex is technically a list with one element.
            filtration= 1 - graph.nodes[v][column]
            )
        # if our census tract is below pop theshold we assign it the value of its highest neighbor. We do this instead of ignoring the tract
        else:
            neighbor_values = [
                graph.nodes[m][column] 
                for m in graph.neighbors(v) if graph.nodes[m][f'{popCol}'] > popMin
            ]
            # if our census tract is an island
            if len(neighbor_values) == 0:
                scomplex.assign_filtration(
                    [v],
                    1
                )
            else: 
                scomplex.assign_filtration(
                    [v],
                    1-max(neighbor_values)
                )
    scomplex.make_filtration_non_decreasing()  
    if dimension == None:  
        persistence = scomplex.persistence()

    else:
        scomplex.compute_persistence()
        persistence = scomplex.persistence_intervals_in_dimension(dimension)

    persistence[np.isinf(persistence)] = 1

    return persistence  

def Homotopy(Column1, Column2, t = 51):
    # INPUT:
        # Two Columns of Data from the same graph
        # The two columns must correspond to the same parts on a map i.e: Column1[0] and Column2[0] are the same census tract
    
    # OUTPUT:
        # Hs -> Straight line Homotopy from Column1 to Column 2
        # Hs[0] is Column1 Hs[1] is Column2

    # Generating t many equal intervals between 0 and 1
    ts = np.linspace(0,1, t)

    # Caluclating change from col1 to col2
    Delta = Column2 - Column1

    Hs = np.array([Column1 + Delta * t for t in ts])

    return Hs

def Get_W_Infinity(PD1, PD2):
    x, matches = gd.hera.wasserstein_distance(PD1,PD2, internal_p = np.inf, matching = True)
    # resetting distances
    dist = []
    for match in matches:
        if match[0] == -1:
            # if the first index of the match is -1, we match the point in PD2 to the diagonal
            dist.append(np.abs((PD2[match[1]][0] - PD2[match[1]][1])/2))
        elif match[1] == -1:
            # if the second index of the match is -1, we match the point in PD1 to the diagonal
            dist.append(np.abs((PD1[match[0]][0] - PD1[match[0]][1])/2))
        else:
            # if neighter are -1, we take the absolute value of the difference between their births's and deaths's
            dist.append(np.max([np.abs((PD1[match[0]][0] - PD2[match[1]][0])),
                              np.abs((PD1[match[0]][1] - PD2[match[1]][1]))]    
                              )) 


    return np.max(dist)

################################################
## Vineyard Functions ##

def stitch(PDs, ts):
    vines = [[0, None, [x,]] for x in range(len(PDs[0]))]
    ends = {x:x for x in range(len(PDs[0]))}
    
    for i in range(1, len(ts)):
        dist, match = gd.hera.wasserstein_distance(PDs[i-1], PDs[i], matching=True)
    
        baby = []
        # print("IIII", i)
    
        new_ends = {k:ends[k] for k in ends}
        for j, (x, y) in enumerate(match):
            # print(x,y)
            if x == -1:
                baby.append(j)
            elif y == -1: # end vines
                vines[ends[x]][1] = i
                vines[ends[x]][2].append(-1)
                # print(f"end {x} -> -1")
            else: # update vines
                vines[ends[x]][2].append(y)
                new_ends[y] = ends[x]
                # print(f"join {x} -> {y} (ind {ends[x]})")
        
        # new vines
        for j in baby:
            x, y = match[j]
            new_ends[y] = len(vines)
            vines.append([i, None, [y,]])
            # print(f"new {y} -> *")
    
        for k in [l for l in ends]: 
            if k >= len(PDs[i]):
                del new_ends[k]
    
        ends = new_ends

    return vines

def vineyard_from_pds(PD0, dim=0, fmax=255, inf=INT_MAX, verbose=False):
    """
    Compute vineyard from precomputed list of persistence diagrams
    Alternative to `vineyard` 
    """
    
    nt = len(PD0)
    ts = np.linspace(0, 1, nt)
    vines = stitch(PD0, ts)

    poss = vines
    
    for i,_ in enumerate(vines):
        # print("II", i, vines[i])
        # print(i, vines[i][2], PD0[0][])
        repl = []
        for j,x in enumerate(vines[i][2]):
            if x == -1:
                # print(PD0[vines[i][0]+j-1][vines[i][2][j-1]], "xx")
                repl.append(np.mean(PD0[vines[i][0]+j-1][vines[i][2][j-1]])*np.ones((2,))) # proj prev
            else:
                repl.append(PD0[vines[i][0]+j][x])
    
        poss[i][2] = np.array(repl)
    
    res = [[[ts[p[0]+np.arange(len(p[2]))][i], *x] for i,x in enumerate(p[2])] for p in poss]

    mxs = [np.max(p[2]) for p in poss if np.inf not in p[2]]
    mx = np.max(mxs) if len(mxs) > 0 else INT_MAX

    if verbose:
        fig, axs = plt.subplots(2, 5, figsize=(20, 8))
       
        inds = [0, round(len(ts)*0.4), round(len(ts)/2), round(len(ts)*0.6), -1]
        for ax_idx, i in enumerate(inds):
            # axs[0, ax_idx].imshow(hs[i])
            axs[0, ax_idx].set_xticks([])
            axs[0, ax_idx].set_yticks([])
            gd.plot_persistence_diagram(PD0[i], axes=axs[1, ax_idx], legend=False)
        plt.tight_layout()
        plt.show()
        
        gos = []
        
        for vine in res:
            vine = np.minimum(np.array(vine), INT_MAX) # INTMAX for INF
            gos.append(go.Scatter3d(x=vine[:,0], y=vine[:,1], z=vine[:,2], marker=dict(
                size=2,
            ),
            line=dict(
                width=2
            )))
        
        xs = np.linspace(0, mx, 10)
        zs = np.linspace(0, 1, 10)
        xss, zss = np.meshgrid(xs, zs)
        gos.append(go.Surface(x=zss, y=xss, z=xss, colorscale=[[0, '#333'], [1, '#333']], opacity=0.1, showscale=False)) # x - y = 0: diag plane
        
        fig = go.Figure(data=gos)
        
        fig.update_layout(
            width=800,
            height=700,
            scene=dict(
              xaxis_title='T (homotopy)',
            zaxis=dict(range=[0,mx], title="Death"),
            yaxis=dict(range=[0,mx], title="Birth")
          ),

        )
        
        fig.show()

    return res

def vineyard(f, g, nt=100, dim=0, fmax=255, inf=INT_MAX, verbose=False):
    ts = np.linspace(0, 1, nt)
    hs = np.array([t*f+(1-t)*g for t in ts]) 
    PD0 = [persistence(h, dimension=dim, fmax=fmax, inf=inf) for h in hs]
    vines = stitch(PD0, ts)

    poss = vines
    
    for i,_ in enumerate(vines):
        # print("II", i, vines[i])
        # print(i, vines[i][2], PD0[0][])
        repl = []
        for j,x in enumerate(vines[i][2]):
            if x == -1:
                # print(PD0[vines[i][0]+j-1][vines[i][2][j-1]], "xx")
                repl.append(np.mean(PD0[vines[i][0]+j-1][vines[i][2][j-1]])*np.ones((2,))) # proj prev
            else:
                repl.append(PD0[vines[i][0]+j][x])
    
        poss[i][2] = np.array(repl)
    
    res = [[[ts[p[0]+np.arange(len(p[2]))][i], *x] for i,x in enumerate(p[2])] for p in poss]

    mxs = [np.max(p[2]) for p in poss if np.inf not in p[2]]
    mx = np.max(mxs) if len(mxs) > 0 else INT_MAX

    if verbose:
        fig, axs = plt.subplots(2, 5, figsize=(20, 8))
       
        inds = [0, round(len(ts)*0.4), round(len(ts)/2), round(len(ts)*0.6), -1]
        for ax_idx, i in enumerate(inds):
            axs[0, ax_idx].imshow(hs[i])
            axs[0, ax_idx].set_xticks([])
            axs[0, ax_idx].set_yticks([])
            gd.plot_persistence_diagram(PD0[i], axes=axs[1, ax_idx], legend=False)
        plt.tight_layout()
        plt.show()
        
        gos = []
        
        for vine in res:
            vine = np.minimum(np.array(vine), INT_MAX) # INTMAX for INF
            gos.append(go.Scatter3d(x=vine[:,0], y=vine[:,1], z=vine[:,2], marker=dict(
                size=2,
            ),
            line=dict(
                width=2
            )))
        
        xs = np.linspace(0, mx, 10)
        zs = np.linspace(0, 1, 10)
        xss, zss = np.meshgrid(xs, zs)
        gos.append(go.Surface(x=zss, y=xss, z=xss, colorscale=[[0, '#333'], [1, '#333']], opacity=0.1, showscale=False)) # x - y = 0: diag plane
        
        fig = go.Figure(data=gos)
        
        fig.update_layout(
            width=800,
            height=700,
            scene=dict(
              xaxis_title='T (homotopy)',
            zaxis=dict(range=[0,mx], title="Death"),
            yaxis=dict(range=[0,mx], title="Birth")
          ),

        )
        
        fig.show()

    return res

def vdist(vines, fD, fL): # diag weight func, length weight func
    V = 0

    for i in range(len(vines)):
        vines[i] = np.minimum(np.array(vines[i]), INT_MAX)
        
    for vine in vines:
        vine = np.array(vine)
        v, L = 0, 0
        for i in range(1, len(vine)):
            l = np.linalg.norm(vine[i][1:]-vine[i-1][1:], ord = np.inf)
            dt = vine[i][0] - vine[i-1][0]

            
            mid = vine[i-1][1:] # first pt ~ mid pt in
            proj = np.mean(mid)*np.ones(2,) # Euclidean projection onto diagonal
            D = np.linalg.norm(proj-mid) # Euclidean dist to diagonal
            w = fD(D)

            ds = l * dt
            v += w * ds 
            L += ds

        v *= fL(L)
        V += v

    return V

## Min Vine Cost ##
def fortyfives(p1,p2):
    # determining direction of 45s
    # whichever has the higher birth time must travel to the left
    
    m1 = -1
    m2 = 1
 
    c1 = np.dot(np.array([-m1,1]), p1.reshape(2,1))
    c2 = np.dot(np.array([-m2,1]), p2.reshape(2,1))
    A = [
        [-m1, 1],
        [-m2, 1],
    ]

    B = [
        c1,
        c2
    ]

    intersect = np.array(np.linalg.solve(A,B))
    path1 = [p1,intersect,p2]

    for i, p in enumerate(path1):
        path1[i] = [ i, p[0][0], p[1][0]]


    m1 = 1
    m2 = -1
 
    c1 = np.dot(np.array([-m1,1]), p1.reshape(2,1))
    c2 = np.dot(np.array([-m2,1]), p2.reshape(2,1))
    A = [
        [-m1, 1],
        [-m2, 1],
    ]

    B = [
        c1,
        c2
    ]

    intersect = np.array(np.linalg.solve(A,B))
    path2 = [p1,intersect,p2]
 

    for i, p in enumerate(path2):
        path2[i] = [ i, p[0][0], p[1][0]]

    return [path1], [path2]

def diag_proj(p1):
    path = []
    x1, y1 = p1
    x2, y2 = ( p1.sum() /2 , p1.sum() / 2)

    path = [[0, x1[0], y1[0]], [1, x2, y2]]
    
    return [path]

def min_vc(PD1, PD2):  
    CM = np.full((PD1.shape[0] + PD2.shape[0], PD1.shape[0] + PD2.shape[0]), np.inf)
    for i, p1 in enumerate(PD1):
        for j, p2 in enumerate(PD2):
            p1 = p1.reshape(-1,1)
            p2 = p2.reshape(-1,1)
            path1, path2 = fortyfives(p1,p2)
            CM[i,j] = np.min([vdist(path1, fD, fL),
                            vdist(path2, fD, fL)
            ])
    # setting PD1 diagonal ells
    for i in range(PD1.shape[0]):
        p = np.array([PD1[i][0], PD1[i][1]]).reshape(-1,1)
        CM[i, i + PD2.shape[0]] = vdist(diag_proj(p),fD,fL)
    # setting PD2 diagonal ells
    for j in range(PD2.shape[0]):
        p = np.array([PD2[j][0], PD2[j][1]]).reshape(-1,1)
        CM[j + PD1.shape[0], j] = vdist(diag_proj(p),fD,fL)
    # setting diagonal matches to zero
    CM[PD1.shape[0]:, PD2.shape[0]:] = 0
    return CM[linear_sum_assignment(CM)].sum()

## Weight Functions ##
def fD(D):
    return D

def fL(L):
    return 1

