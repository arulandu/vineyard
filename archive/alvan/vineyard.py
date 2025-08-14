import sklearn
from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly
import plotly.graph_objects as go

import gudhi
import gudhi.wasserstein
import gudhi.hera
import ot

INT_MAX = 2147483647

def persistence(array, fmax=255., dimension=None):
    height, width = array.shape
    cubeComplex = gudhi.CubicalComplex(
        dimensions = [width,height],
        top_dimensional_cells = fmax - array.flatten()
    )
 
    if dimension == None:
        persistence = cubeComplex.persistence()
    else:
        cubeComplex.compute_persistence()
        persistence = cubeComplex.persistence_intervals_in_dimension(dimension)
        persistence[np.isinf(persistence)] = fmax

    return persistence

def stitch(PDs, ts, method=gudhi.hera):
    vines = [[0, None, [x,]] for x in range(len(PDs[0]))]
    ends = {x:x for x in range(len(PDs[0]))}
    
    for i in range(1, len(ts)):
        dist, match = method.wasserstein_distance(PDs[i-1], PDs[i], matching=True)
    
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

def make_pds(f, g, ts, fmax=255.0, dim=1):
    return [persistence((1-t)*f+t*g, fmax=fmax, dimension=dim) for t in ts]

def vineyard(ts, PD0, method=gudhi.hera, verbose=False):
    vines = stitch(PD0, ts, method=method)

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

    if verbose > 0:
        bnds = [[np.max(p[2]), np.min(p[2])] for p in poss if np.inf not in p[2]]
        mx, mn = (np.max([p[0] for p in bnds]), np.min([p[1] for p in bnds])) if len(bnds) > 0 else (INT_MAX, -INT_MAX)

        if verbose == 1 or verbose == 3:
            gudhi.plot_persistence_diagram(PD0[0])
            plt.show()
            gudhi.plot_persistence_diagram(PD0[-1])
            plt.show()
        
        if verbose == 2 or verbose == 3:
            gos = []
            
            for vine in res:
                vine = np.array(vine)
                # vine = np.minimum(np.array(vine), INT_MAX) # UNCOMMENT TO PLOT INF LINE
                # print(vine)
                gos.append(go.Scatter3d(x=vine[:,0], y=vine[:,1], z=vine[:,2], marker=dict(
                    size=2,
                ),
                line=dict(
                    width=2
                )))
            
            xs = np.linspace(min(0, mn), mx, 10)
            zs = np.linspace(np.min(ts), np.max(ts), 10)
            xss, zss = np.meshgrid(xs, zs)
            gos.append(go.Surface(x=zss, y=xss, z=xss, colorscale=[[0, '#333'], [1, '#333']], opacity=0.1, showscale=False)) # x - y = 0: diag plane
            
            fig = go.Figure(data=gos)
            
            fig.update_layout(
                width=800,
                height=700,
                scene=dict(
                xaxis_title='T (homotopy)',
                yaxis_title='Birth',
                zaxis_title='Death'
            )
            )
            
            fig.show()

    return res

def vdist(vines, fD=lambda _:1, fL=lambda _:1): # diag weight func, length weight func
    V = 0

    for i in range(len(vines)):
        vines[i] = np.minimum(np.array(vines[i]), INT_MAX)

    for vine in vines:
        vine = np.array(vine)
    
        v, L = 0, 0
        for i in range(1, len(vine)):
            l = np.linalg.norm(vine[i][1:]-vine[i-1][1:], ord=np.inf)
            dt = vine[i][0] - vine[i-1][0]
    
            mid = np.mean([vine[i][1:],vine[i-1][1:]], axis=0)
            D = (mid[1]-mid[0])/2
            
            v += fD(D)*l
            L += l*dt
    
        v *= fL(L)
        V += v

    return V