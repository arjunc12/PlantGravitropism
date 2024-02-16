import numpy as np
import networkx as nx
from Networkx_Conversion import *
from Wiring_cost_conduction_delay import *
from Distances import *

#find best G and alpha using point differnce
    #input: networx arbor
    #output:
        #closest optimal networkx arbor
        #Closest G
        #closest alpha
def PointbestDiffernce(Plant):
    Grange = np.arange(-2, 3, 1)
    bestG = -2
    aRange = np.arange(0,1.1,.1)
    bestA =0
    Plant2 = optimalArbor(Plant, bestG, bestA)
    bestDif = pointDif(Plant,Plant2)
    for G in Grange:
        for alpha in aRange:
            Plant2 = optimalArbor(Plant, G, alpha)
            dif = findDiffernce(Plant,Plant2,bestDif)
            if dif < bestDif:
                bestDif = dif
                bestA = alpha
                bestG =G
        print(G)#for me to see if it is still working
    bestPlant = optimalArbor(Plant,bestG, bestA)
    return(bestPlant,bestG, bestA)

#saves a little time in that it bales if the current dif is too large
#basicly the same as pointDif
def findDiffernce(PlantOg,PlantNew,bestDif):
    dif =0.0
    for node in PlantOg.nodes():
        if PlantOg.nodes[node]['label'] != 'lateral root':
            if PlantOg.nodes[node]['label'] != 'lateral root tip':
                if PlantOg.nodes[node]['label'] != 'lateral root base':
                    continue
        rootNumber = PlantOg.nodes[node]['root number']
        if rootNumber ==0:
            continue
        ox,oy = PlantOg.nodes[node]['coordinate']
        for edge in PlantNew.edges():
            if PlantNew.edges[edge]['label'] != 'lateral root':
                continue
            elif PlantNew.edges[edge]['root number'] != rootNumber:
                continue
            nxs = PlantNew.edges[edge]['xs']
            nys = PlantNew.edges[edge]['ys']
            bestDx = np.absolute(nxs[0]-ox)
            besti = 0
            for i in range(1, len(nxs)):
                dx = np.absolute(nxs[i] -ox)
                if dx < bestDx:
                    bestDx = dx
                    besti = i
            dy = np.absolute(nys[besti] -oy)
            dif = dif + dy 
            if dif > bestDif:
                break
        if dif> bestDif:
            break
    return(dif)

#finds best G and alpha using point diffence
    #input: netowrkx arbor with the connections to mainroot
    #output:
        #front: networkx object of all nodes with coordinates (conduction delay, wiring cost) for diffent Gs and alphas
        #orignal plant
        #the best euclid distance
        #bestNodeArray
            #the cordinate which is (conducton delay, wiringcost)
            #the best G value
            #best alpha value
        #CD: conduction delay
        #WC:wiring cost
def ClosestEuclid(originalPlant):
    Grange = np.arange(-2, 3, 1)
    Arange = np.arange(0,1.1,.1)
    front = nx.Graph()
    cds=[]
    wcs=[]
    c =0
    WC = calculateWC(originalPlant)
    CD = calculateCD(originalPlant)
    bestDis = 10**20000
    for G in Grange:
        print(G) #just to see how long it takes
        for a in Arange:
            newplant = optimalArbor(originalPlant, G, a)
            wc =calculateWC(newplant)
            cd = calculateCD(newplant)
            cds.append(cd)
            wcs.append(wc)
            front.add_node(c)
            front._node[c]['coordinate']=(cd,wc)
            front._node[c]['G'] =G
            front._node[c]['alpha'] =a
            c+=1
            dis = np.sqrt((x-cd)**2 +(y-wc)**2)
            if dis < bestDis:
                bestDis = dis
                bestNode = c
    bestNodeArray = [front._node[bestNode]['coordinate'], front._node[bestNode]['G'],front._node[bestNode]['alpha']]
    return(front, originalPlant, bestDis,bestNodeArray,CD,WC)
    #front: networkx object of all nodes with coordinates (conduction delay, wiring cost) for diffent Gs and alphas
    #orignal plant
    #the best euclid distance
    #bestNodeArray
        #the cordinate which is (conducton delay, wiringcost)
        #the best G value
        #best alpha value
    #CD: conduction delay
    #WC:wiring cost

#Euclid distance version 2 the better one

#Use ClosEuclid(ogPlant,front) to find best G and alpha using Eucliding distance given a pareto front
    #input:
        #ogPlant: networkx object plant arbor with connected latteral roots
        #front: networx object that is the paeto front
    #output:
        # G is the best G value
        # alpha is the best alpha value
        # bestDis the distance for the Euclidan distsnce
        #bestCD the conduction delay value that is the closest
        #bestWC the wiring cost value that is the closest
def ClosEuclid(ogPlant,front):
    WC = calculateWC(ogPlant)
    CD = calculateCD(ogPlant)
    bestDis = 10**20000
    for node in front.nodes():
        cd,wc = front.nodes[node]['coordinate']
        dis = np.sqrt((CD-cd)**2 +(WC-wc)**2)
        if dis < bestDis:
            bestDis = dis
            G = front.nodes[node]['G']
            alpha = front.nodes[node]['alpha']
            bestCD, bestWC = front.nodes[node]['coordinate']
    return(G,alpha,bestDis, bestCD, bestWC)

#Scaling distance

#Use ClosestScaling(ogPlant,front) to find the best G and alpha using scaling distance method given a pareto fron
     #input:
        #ogPlant: networkx object plant arbor with connected latteral roots
        #front: networx object that is the paeto front
     #output:
        # G is the best G value
        # alpha is the best alpha value
        # bestDis the distance for the Euclidan distsnce
        #bestCD the conduction delay value that is the closest
        #bestWC the wiring cost value that is the closest
def ClosestScaling(ogPlant,front):
    WC = calculateWC(ogPlant)
    CD = calculateCD(ogPlant)
    bestDis = 10**20000
    for node in front.nodes():
        cd,wc = front.nodes[node]['coordinate']
        dis = max((WC/wc),(CD/cd))
        if dis < bestDis:
            bestDis = dis
            G = front.nodes[node]['G']
            alpha = front.nodes[node]['alpha']
            bestCD, bestWC = front.nodes[node]['coordinate']
    return (G,alpha, bestDis,bestCD, bestWC)

# now both making a front and finding the best point differnce

#Use PointbestDiffernceAndFront(Plant) to find best G and alpha using point differnce
    #input: networx arbor
    #output:
        #closest optimal networkx arbor using point diffence method
        #Closest G
        #closest alpha
        #pareto front as Networkx object
    #this way you can kill two birds with one stone 
        #ie contsruct the peato front while looking for the best arbor using the point differnce method
def PointbestDiffernceAndFront(Plant):
    Grange = np.arange(-2, 3, 1)
    bestG = -2
    aRange = np.arange(0,1.1,.1)
    bestA =0
    front = nx.Graph()
    c=0
    Plant2 = optimalArbor(Plant, bestG, bestA)
    cd =calculateCD(Plant2)
    wc = calculateWC(Plant2)
    bestCD, bestWC = cd, wc
    front.add_node(c)
    front._node[c]['coordinate']=(cd,wc)
    front._node[c]['G'] =bestG
    front._node[c]['alpha'] =bestA
    c+=1
    bestDif = pointDif(Plant,Plant2)
    for G in Grange:
        for alpha in aRange:
            Plant2 = optimalArbor(Plant, G, alpha)
            cd =calculateCD(Plant2)
            wc = calculateWC(Plant2)
            front.add_node(c)
            front._node[c]['coordinate']=(cd,wc)
            front._node[c]['G'] =G
            front._node[c]['alpha'] =alpha
            c+=1
            dif = findDiffernce(Plant,Plant2,bestDif)
            if dif < bestDif:
                bestDif = dif
                bestA = alpha
                bestG =G
                bestCD, bestWC = cd, wc
        print(G)#for me to see if it is still working
    bestPlant = optimalArbor(Plant,bestG, bestA)
    return(bestPlant,bestG, bestA, front, bestCD, bestWC)

