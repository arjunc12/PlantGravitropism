#import matplotlib.pyplot as plt
import numpy as np
import pylab
import scipy
import plotly
from plotly import graph_objs as go
from plotly.offline import iplot, plot, init_notebook_mode
import networkx as nx

#takes file
#returns usable arrays of main root points and pq points
def parseData(file):
    import csv
    Data =[]
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for lines in csv_reader:
           Data.append(lines)
        mainRootPoints =[]
        c=1
        while len(Data[c]) != 1:
            point = [float(Data[c][0]),float(Data[c][1])]
            mainRootPoints.append(point)
            c= c+1
        pqs =[]
        pq=[]
        c=c+1
        while c < len(Data):
            if len(Data[c]) ==1:
                pqs.append(pq)
                pq =[]
            if len(Data[c]) ==2:
                point = [float(Data[c][0]),float(Data[c][1])]
                pq.append(point)
            c = c+1
        pqs.append(pq)
    return (mainRootPoints, pqs)
#Making the data into readable segments

#uses parseData
#takes a file name 
#returns networkx object 
    #Object does not have lateral roots connected to main root!!
def readArborFile(fileName):
    Array = parseData(fileName)
    mainRootPoints = Array[0]
    lateralRoots =Array[1]
    Plant = nx.Graph()
    #adding the mainroot base to graph
    Plant.add_node(0)
    Plant._node[0]['label']= 'main root base'
    Plant._node[0]['coordinate'] = [mainRootPoints[0][0], mainRootPoints[0][1]]
    Plant._node[0]['root number']= 0
    #constructing the main root
    for i in range(1, len(mainRootPoints)):
        Plant.add_node(i)
        Plant._node[i]['label']= 'main root'
        Plant._node[i]['coordinate'] = (mainRootPoints[i][0], mainRootPoints[i][1])
        Plant._node[i]['root number']=0
        Plant.add_edge(i-1,i)
        length = np.sqrt((Plant._node[i]['coordinate'][0] - Plant._node[i-1]['coordinate'][0])**2 +(Plant._node[i]['coordinate'][1] -Plant._node[i-1]['coordinate'][1])**2)
        Plant[i-1][i]['length'] = length
        Plant[i-1][i]['label'] = 'main root'
        Plant[i-1][i]['G'] = 0
        Plant[i-1][i]['root number']=0
    #this way we can have individual labels for points
    count = len(mainRootPoints)
    rootNumber =1
    #constructing the lateral roots
    for root in lateralRoots:
        # constructing a lateral root
        #making the first point
        Plant.add_node(count)
        Plant._node[count]['label'] = 'lateral root base'
        Plant._node[count]['coordinate'] = (root[0][0],root[0][1])
        Plant._node[count]['root number'] = rootNumber
        #making the inside points and edges
        for i in range(1,len(root) -1):
            c =count +i
            Plant.add_node(c)
            Plant._node[c]['label'] = 'lateral root'
            Plant._node[c]['coordinate'] = (root[i][0],root[i][1])
            Plant._node[c]['root number'] = rootNumber
            Plant.add_edge(c-1,c)
            length = np.sqrt((Plant._node[c]['coordinate'][0] - Plant._node[c-1]['coordinate'][0])**2 +(Plant._node[c]['coordinate'][1] -Plant._node[c-1]['coordinate'][1])**2)
            Plant[c-1][c]['length']= length
            Plant[c-1][c]['label'] = 'lateral root'
            Plant[c-1][c]['G']=0
            Plant[c-1][c]['root number'] = rootNumber
        #making the root tip
        count = count +len(root)-1
        Plant.add_node(count)
        Plant._node[count]['label'] = 'lateral root tip'
        Plant._node[count]['coordinate'] = (root[-1][0],root[-1][1])
        Plant._node[count]['root number'] = rootNumber
        Plant.add_edge(count-1,count)
        length = np.sqrt((Plant._node[count]['coordinate'][0] - Plant._node[count-1]['coordinate'][0])**2 +(Plant._node[count]['coordinate'][1] -Plant._node[count -1]['coordinate'][1])**2)
        Plant[count-1][count]['length']= length
        Plant[count-1][count]['label'] = 'lateral root'
        Plant[count-1][count]['G'] =0
        Plant[count-1][count]['root number'] = rootNumber
        count = count+1
        rootNumber = rootNumber+1
    # return a networkX graph of the arbor
    return Plant

#takes networkx object as input (the given file arbor)
#returns network x object 
    # now the latteral roots are connected to mainroot
def findMainRootConnections(plant):
    mainRootNodes =[]
    c=0
    #sets the new plant to equal the old so we dont have to waist time adding the nodes and edges we already have
    Plant1= plant
    rootBases =[]
    #sorts the nodes into mainroot, and lateral root bases
    for node in plant.nodes():
        c = c+1
        if plant.nodes[node]['label'] == 'main root':
            mainRootNodes.append(node)
        elif plant.nodes[node]['label'] == 'main root base':
            mainRootNodes.append(node)
        elif plant.nodes[node]['label'] == 'lateral root base':
            rootBases.append(node)
        else:
            continue 
    #for each latteral root base this finds the segment of main root in wich one of the 
    #main root points y values is less than the root base y value and the other main root point y value
    #is greater then the root base y value.
    #then it finds where it should connect on that segment and creates a new node and edge
    #it also finds the length of the new edge
    for rootBase in rootBases:
        if plant.nodes[rootBase]['label'] != 'lateral root base':
            continue
        rx,ry = plant.nodes[rootBase]['coordinate']
        main1x, main1y =0,0
        main2x, main2y =plant.nodes[mainRootNodes[0]]['coordinate']
        for i in range(1, len(mainRootNodes)-1):
            m1x,m1y = plant.nodes[mainRootNodes[i]]['coordinate']
            m2x, m2y = plant.nodes[mainRootNodes[i+1]]['coordinate']
            if m1y <= ry:
                if m2y >= ry:
                    main1x,main1y = m1x,m1y
                    main2x, main2y =m2x,m2y
                    break                
        rootNumber = plant.nodes[rootBase]['root number']
        Plant1.add_node(c)
        Plant1._node[c]['label'] = 'connection'
        m = (main2y-main1y)/(main2x-main1x)
        x = ((ry -main2y)/m)+main2x
        Plant1._node[c]['coordinate'] = (x,ry)
        Plant1._node[c]['root number'] = rootNumber
        Plant1.add_edge(c, rootBase)
        length = np.sqrt((plant._node[c]['coordinate'][0] - plant._node[rootBase]['coordinate'][0])**2 +(plant._node[c]['coordinate'][1] -plant._node[rootBase]['coordinate'][1])**2)
        Plant1[c][rootBase]['length'] = length
        Plant1[c][rootBase]['root number'] = rootNumber
        Plant1[c][rootBase]['G'] = 0
        Plant1[c][rootBase]['label'] = 'lateral root'
        c=c+1
    return(Plant1)
# takes given networx arbor as input 
#returns given networx arbor with connections to main root

#takes file name as input
# returns networx arbor of given data with lateral roots connected to mainroot
def connectedArbor(file):
    plant = readArborFile(file)
    plant1 = findMainRootConnections(plant)
    return (plant1)

#Now to find the optimal arbor

#Takes in networx arbor, G, alpha
#returns the optimal arbor as a networx graph
def optimalArbor(arbor, G, alpha):
    MainRootPoints = []
    pqs=[]
    Plant = nx.Graph()
    c=0
    #adds each relivent nodes and edges from the given arbor
    for node in arbor.nodes():
        x, y = arbor.nodes[node]['coordinate']
        #makes the main root base
        if 'main root base' == arbor.nodes[node]['label']:
            MainRootPoints.append([x,y])
            Plant.add_node(c)
            Plant._node[c]['label'] = arbor.nodes[node]['label']
            Plant._node[c]['coordinate'] = arbor.nodes[node]['coordinate']
            Plant._node[c]['root number'] =arbor.nodes[node]['root number']
            c = c+1
        #makes the main root nodes and edges and calculates length of edges
        elif 'main root' == arbor.nodes[node]['label']:
            MainRootPoints.append([x,y])
            Plant.add_node(c)
            Plant._node[c]['label'] = arbor.nodes[node]['label']
            Plant._node[c]['coordinate'] = arbor.nodes[node]['coordinate']
            Plant.add_edge(c-1,c)
            length = np.sqrt((Plant._node[c]['coordinate'][0] - Plant._node[c-1]['coordinate'][0])**2 +(Plant._node[c]['coordinate'][1] -Plant._node[c-1]['coordinate'][1])**2)
            Plant[c-1][c]['length'] = length
            Plant[c-1][c]['label'] = 'main root'
            Plant[c-1][c]['G'] =0
            Plant[c-1][c]['root number'] =0
            c = c+1
        #since we only really care about the lateral root tips, we will keep them 
        elif 'lateral root tip' == arbor.nodes[node]['label']:
            pqs.append([x,y])
            Plant.add_node(c)
            Plant._node[c]['label'] = arbor.nodes[node]['label']
            Plant._node[c]['coordinate'] = arbor.nodes[node]['coordinate']
            c = c+1
    #main root is root number one so the lateral roots start with 1
    rootNumber =1
    #for each latteral root tip
    for pq in pqs:
        Array =findMostOptimal(MainRootPoints,G, alpha,pq) 
        #findMostOptimal takes in an array of the main root points, G, alpha, and the current latteral root tip
        #returen [txy,xs,ys, cd, length]
        #txy: is the optimal connection to main root
        #xs, ys: are the points along the curve
        #cd is conduction delay
        #length: is the length of the cuved line
        length = Array[4]
        txy =Array[0]
        xs=Array[1]
        ys=Array[2]
        Plant.add_node(c)
        Plant._node[c]['label']='connection'
        Plant._node[c]['root number'] = rootNumber
        tx = txy[0][0]
        ty = txy[1][0]
        Plant._node[c]['coordinate'] = [tx,ty]
        Plant._node[c]['pq'] = pq
        #Now we make the edge that connects the lateral root tip to optimal point to the mainroot
        for node in Plant.nodes():
            #once we find the right lateral root tip we break the for loop
            if 'lateral root tip' == Plant.nodes[node]['label']:
                if (pq[0],pq[1]) == Plant.nodes[node]['coordinate']:
                    Plant.add_edge(c, node)
                    Plant[c][node]['length'] = length
                    Plant[c][node]['label'] ='lateral root'
                    Plant[c][node]['root number'] = rootNumber
                    Plant[c][node]['G'] = G
                    #we store the xs and ys so that when we graph this object we don't have to find the curve twice
                    Plant[c][node]['xs'] = xs
                    Plant[c][node]['ys']= ys
                    break
        rootNumber = rootNumber +1
        c=c+1
    return Plant
    # arbor is a networkx graph
    # return a networkx graph representing optimal arbor

#used in optimalArbor
#takes the array of mainroot points, G, al is alpha, and a pq coordinate
#returns:
    #optimal connection to the mainroot
    #the xs and ys of the curved line
    #conduction delay
    #length of the curve
def findMostOptimal(MainRootPoints,G, al,pq):
    p = pq[0]
    q = pq[1]
    point1 = MainRootPoints[0]
    point2 = MainRootPoints[1]
    CD = 0
    #best length is actually the best fit for the alpha given
    bestDAC = findOptimalInSegment(point1, point2,G,al,pq,CD) 
    #return length, txy, xs,ys,cd, actual len
    bestLength = bestDAC[0]
    #for each segment comapre the alpha score to the best current alphascore
    for i in range(2, len(MainRootPoints)):
        length = findOptimalInSegment(point1, point2,G,al,pq,CD)[0]
        if length < bestLength:
            bestDAC = findOptimalInSegment(point1, point2,G,al,pq,CD)
            #returns length, txy, xs, ys, cd, actual len
            bestLength = length
        #increase conduction delay
        CD = CD +findOptimalInSegment(point1, point2,G,al,pq,CD)[4] 
        point1 = point2
        point2 = MainRootPoints[i]
    return (bestDAC[1:])
    #returen [txy,xs,ys, cd, length]
        #txy: the optimal point in wich latteral root connects to main root
        #xs,ys: the points along the curve
        #cd conduction delay
        #length: length of curve

#used in find most optimal to comapare main root segments
#this finds the best connection for each segment
#input:
    #two points that make up the main root
    #G, and al: alpha
    # pq: cordinate for latteral root tip
    #conduction delay
#returns:
    #the alpha score
    #best connection to main root in this segment
    #xs,ys: points along the curve
    #conductin delay
    #length of curve
def findOptimalInSegment(point1, point2,G,al,pq,CD):
    a =point1[0]
    b= point1[1]
    c= point2[0]
    d = point2[1]
    p=pq[0]
    q=pq[1]
    theta = findTheta (a,b,c,d) 
    ArrayDAC = deAngleCurve(G,al,p,q,theta,a,b,c,d,CD) 
    #return length, txy, xs,ys, cd, actual len
    return (ArrayDAC)
    #return length, txy, xs,ys,cd, actual len

# used in findOptimalInSegment
#takes the mainroot points as (a,b),(c,d) to then find the theta in wich to rotate the graph
def findTheta (a,b,c,d):
    theta = np.arctan((np.absolute(a-c))/(np.absolute(b-d)))
    m = (b-d)/(a-c)
    if m > 0:
        theta =-theta
    return theta

#Used in findOptimalInSegment
#takes a angled mainroot
#rotates the angled mainroot by theata
#finds the optimal connection 
#then reanlges the root with the new connection
#input:
    #G, al: alpha
    #p,q the oringal lateral root tip
    #theta: the angle to rotate the line
    #main root points (a,b), (c,d)
    #CD: condution delay
#returns:
    #alpha score
    #optimal connection to main root
    #xs,ys: points along the curve
    #conduction delay
    #length of curve
def deAngleCurve(G,al,p,q,theta,a,b,c,d,CD):
    pq = rotateLine([p],[q], theta, c,d) 
    #returns pq in this case
    pn = pq[0][0]
    qn = pq[1][0]
    tmax = rotateLine([c],[d], theta, a,b)
    tmax = tmax[1][0]
    best = None
    if al == 0:
        best = findBest(tmax, G,pn,qn,al,CD)
    else:
        best = findBestRichards(tmax, G,pn,qn,al,CD)
    #returns best length and t, actulal lenth
    t = best[1]
    actlen =best[2]
    XYPrime = XYDeAngle (tmax,G,pn,qn,t)
    #return x,y
    txy = reAngle([0],[best[1]], theta,c,d)
    XY = reAngle(XYPrime[0], XYPrime[1], theta,c,d)
    return(best[0], txy,XY[0],XY[1],CD, actlen)
    #return length, txy, xs,ys, conducton delay, actual length

#used in deAngleCurve
#rotates a line or point by theta
#input:
    #xs, ys: the points you want to rotate
    #theta: angle in wich you rotate
    #(a,b): one of the main root points
#returns:
    #now rotated points
def rotateLine(xs,ys,theta,a,b):
    nx=[]
    ny=[]
    R = [[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]
    for i in range(0,len(xs)):
        A =[xs[i]-a,ys[i]-b]
        xy=np.array(np.matmul(A,R))
        nx.append(xy[0])
        ny.append(xy[1])
    return(nx, ny)

#used in deAngleCurve
# finds the best connection on a vetical mainroot
#input:
    #Tmax: the length of the main root segment 
    #G
    #(p,q) the new lateral root tip now that we have deganled it
    #a : is alpha
    #CD: conduction delay
#returns:
    #the alpha score
    #where it connects
    #length of curve
def findBest(Tmax,G,p,q,a,CD):
    t=0
    Tmax = -Tmax
    bestLength= 100000000000000000000.0
    bestT = t
    actLen =bestLength
    while t >= Tmax:
        b = (q - G * (p**2) - t)/p
        if p>0:
            x =np.linspace(0, p, num=50)
        elif p < 0:
            x =np.linspace(p, 0, num=50)
        lot = np.sqrt(1+(2*G*x + ((q-G*(p**2)-t)/p))**2)
        lot = scipy.integrate.trapezoid(lot,x)
        y= (1 - a) *(lot) + a *(lot+CD+t)
        length = y
        if length < bestLength:
            bestLength = length
            bestT = t
            actLen=lot
        #t = t - 0.001 #last
        t = t - 0.01 #last 
    return(bestLength,bestT,actLen)

#used in deAngleCurve
# finds the best connection on a vetical mainroot
#input:
    #Tmax: the length of the main root segment 
    #G
    #(p,q) the new lateral root tip now that we have deganled it
    #a : is alpha
    #CD: conduction delay
#returns:
    #the alpha score
    #where it connects
    #length of curve
def findBestRichards(Tmax,G,p,q,a,CD):
    t=0
    Tmax = -Tmax
    bestLength= float("inf")
    bestT = t
    bestTCandidates = [0, Tmax]
    assert a != 0
    radical_inside = ((G ** 2) * (p ** 2)) + (1.0 / ((2 * a) - (a ** 2)))
    radical = radical_inside ** 0.5
    sol1 = (-G * p) + (1 - a) * radical
    sol2 = (-G * p) - (1 - a) * radical
    if sol1 <= 0 and sol1 >= Tmax:
        bestTCandidates.append(sol1)
    if sol2 <= 0 and sol2 >= Tmax:
        bestTCandidates.append(sol2)
    actLen =bestLength
    for t in bestTCandidates:
        b = (q - G * (p**2) - t)/p
        if p>0:
            x =np.linspace(0, p, num=50)
        elif p < 0:
            x =np.linspace(p, 0, num=50)
        lot = np.sqrt(1+(2*G*x + ((q-G*(p**2)-t)/p))**2)
        lot = scipy.integrate.trapezoid(lot,x)
        y= (1 - a) *(lot) + a *(lot+CD+t)
        length = y
        if length < bestLength:
            bestLength = length
            bestT = t
            actLen=lot
        #t = t - 0.001 #last
        t = t - 0.01 #last 
    return(bestLength,bestT,actLen)

#used in deAngleCurve
#this creates the curved line in the deAnged space
#input:
    #tmax: the lenght of the main root
    #G
    #(p,q): the deAngled latteral root tip
    #t: the y value of the best connection to deAngled main root
#return:
    #the xs, and ys of the deAngled curve
def XYDeAngle (tmax,G,p,q,t):
    x = np.linspace(0,p)
    if p <0:
        x = np.linspace(p,0)
    b=(q - G*(p**2) -t)/p
    y=G * x**2 + b*x + t
    return (x,y)

#used in deAngleCurve
# rotates a line or point back
#input:
    #(xs,ys) the points in wich you want to rotate back
    #theta: the origal angle you rotated
    #(a,b) point on the orignal main root segment
#return:
    #the reAngled points
def reAngle(xs, ys, theta, a,b):
    nx =[]
    ny=[]
    R = [[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]
    inR = np.linalg.inv(R)
    ab= [a,b]
    for i in range(0, len(xs)):
        Ap = [xs[i],ys[i]]
        App = np.matmul(Ap,inR)
        xy=np.array(np.add(App,ab))
        nx.append(xy[0])
        ny.append(xy[1])
    return(nx, ny)