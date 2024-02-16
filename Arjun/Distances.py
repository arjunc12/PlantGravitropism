#Use euclidDis(plant1, plant2) to calculate euclidean distance of cost vectors
    #input: two networx arbors
    #output: euclidian distance of the cost vectors
    
#Use pointDif(plant1,plant2) to calculate point differnce of two arbors
    #input: two networx arbors
    #output: point differnce of the two plants

import numpy as np
import os
from Wiring_cost_conduction_delay import *

#calculates the euclidean distance of cost vectors
    #input: two networx arbors
    #output: euclidian distance of the cost vectors
def euclidDis(plant1, plant2):
    wc1 = calculateWC(plant1)
    cd1 = calculateCD(plant1)
    wc2 = calculateWC(plant2)
    cd2 = calculateCD(plant2)
    euDis = np.sqrt((wc1-wc2)**2 +(cd1-cd2)**2)
    return(euDis)

#calculates the point differnce of two arbors
    #input: two networx arbors
    #output: point differnce of the two plants
def pointDif(plant1,plant2):
    diffence =0.0
    for node in plant1.nodes():
        if plant1.nodes[node]['label'] != 'lateral root':
            if plant1.nodes[node]['label'] != 'lateral root tip':
                continue
        rootNumber = plant1.nodes[node]['root number']
        if rootNumber ==0:
            continue
        ox,oy = plant1.nodes[node]['coordinate']
        for edge in plant2.edges():
            if plant2.edges[edge]['label'] != 'lateral root':
                continue
            nxs = plant2.edges[edge]['xs']
            nys = plant2.edges[edge]['ys']
            bestDx = np.absolute(nxs[0]-ox)
            besti = 0
            for i in range(1, len(nxs)):
                dx = np.absolute(nxs[besti] -ox)
                if dx < bestDx:
                    bestDx = dx
                    besti = i
            #dy = np.absolute(nys[i] -oy)
            dy = (nys[i] -oy) ** 2
            diffence = diffence +dy 
    return(diffence)

