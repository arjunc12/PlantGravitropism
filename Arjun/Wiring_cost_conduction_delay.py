#Use calculateWC(plant) to calculate wirng cost
## takes networx object as input
### make sure the object has lateral roots connected to main root

# Use calculateCD(plant) to calculate wiring cost
##takes networx object as input
### make sure the object has lateral roots connected to main root

#Will work for both optimal and given arbors

import numpy as np

#takes networx arbor  
#returns calculates wiring cost
def calculateWC(plant):
    wc =0
    for edge in plant.edges():
        wc = wc + plant.edges[edge]['length']
    return(wc)

#takes networkx arbor
#returns conduction delay
def calculateCD(plant):
    mainRootNodes =[]
    numOfLatRoots =0
    for node in plant.nodes():
        if plant.nodes[node]['label'] == 'main root':
            mainRootNodes.append(node)
        elif plant.nodes[node]['label'] == 'main root base':
            mainRootNodes.append(node)
        elif plant.nodes[node]['label'] == 'connection':
            if numOfLatRoots < plant.nodes[node]['root number']:
                numOfLatRoots = plant.nodes[node]['root number']
    cd =0
    for i in range (1, numOfLatRoots +1):
        cdForRoot =0
        for edge in plant.edges():
            if plant.edges[edge]['root number'] != i:
                continue
            cdForRoot += plant.edges[edge]['length']
        for node in plant.nodes():
            if plant.nodes[node]['label']!= 'connection':
                continue
            if plant.nodes[node]['root number'] == i:
                cx,cy = plant.nodes[node]['coordinate']
                break
        countMainRoot = False
        for j in range(0,len(mainRootNodes)-1):
            if countMainRoot:
                cdForRoot += plant[mainRootNodes[j]][mainRootNodes[j+1]]['length']
                continue
            m1x,m1y = plant.nodes[mainRootNodes[j]]['coordinate']
            m2x, m2y = plant.nodes[mainRootNodes[j+1]]['coordinate']
            if m1y <= cy:
                if m2y >= cy:
                    countMainRoot =True
                    length = np.sqrt((m2x -cx)**2 +(m2y-cy)**2)
                    cdForRoot += length
                    continue 
        cd +=cdForRoot
    return(cd)  
#takes networkx arbor
#returns conduction delay


