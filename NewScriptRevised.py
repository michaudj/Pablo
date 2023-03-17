# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 20:23:27 2021

@author: jermi792
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 10:28:48 2021

@author: jermi792
"""
import os
os.environ["PROJ_LIB"] = "C:\\Users\\jermi792\\.julia\\v0.6\\Conda\\deps\\usr\\envs\\py36\\Library\\share\\basemap";
#import conda
#
#conda_file_dir = conda.__file__
#conda_dir = conda_file_dir.split('lib')[0]
#proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
#os.environ["PROJ_LIB"] = proj_lib

import numpy as np
import pandas as pd
import math
import community
import networkx as nx

import pysal
from pysal.explore.esda import Gamma

## function to create a weight matrix from a shapefile...
#pysal.lib.weights.Rook.from_shapefile




from scipy import spatial
from scipy.stats import entropy

import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.cm
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize, LogNorm



def js(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
   # normalize
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2


def bhattacharyya(a, b):
    """ Bhattacharyya distance between distributions (lists of floats). """
    if not len(a) == len(b):
        raise ValueError("a and b must be of the same size")
    return 1-sum((math.sqrt(u * w) for u, w in zip(a, b)))

def mapDistance(map1, map2):
    count = 0.
    for i in map1:
        if map1[i] != map2[i]:
            count += 1
    return count/len(map1)


def confusionMatrix(Com1,Com2):
    n1 = len(Com1)
    n2 = len(Com2)
    ConfMat = np.zeros((n1,n2),int)
    for i in range(n1):
        for j in range(n2):
            ConfMat[i,j] = len(set(Com1[i]) & set(Com2[j]))
    return ConfMat

def confusionMatrix2(Com1,Com2):
    n1 = len(Com1)
    n2 = len(Com2)
    ConfMat = np.zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            ConfMat[i,j] = len(set(Com1[i]) & set(Com2[j]))/float(len(set(Com1[i])))
    return ConfMat

def confusionMatrix3(Com1,Com2):
    n1 = len(Com1)
    n2 = len(Com2)
    ConfMat = np.zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            ConfMat[i,j] = len(set(Com1[i]) & set(Com2[j]))/float(len(set(Com2[j])))
    return ConfMat

def confusionMatrix4(Com1,Com2):
    n1 = len(Com1)
    n2 = len(Com2)
    ConfMat = np.zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            ConfMat[i,j] = 2*len(set(Com1[i]) & set(Com2[j]))/float(len(set(Com2[j]))+len(set(Com1[i])))
    return ConfMat

def Criteria(Com1,Com2):
    return (confusionMatrix2(Com1,Com2) + confusionMatrix2(Com1,Com2))/2

def JaccardMatrix(Com1,Com2):
    n1 = len(Com1)
    n2 = len(Com2)
    ConfMat = np.zeros((n1,n2))
    for i in range(n1):
        for j in range(n2):
            ConfMat[i,j] = len(set(Com1[i]) & set(Com2[j]))/len(set(Com1[i]) | set(Com2[j]))
    return ConfMat

def AverageVoteShareInCom(Commun,y,measure,Voteshares,index):
    VoteshareAverage = 0 * Voteshares[y][index]
    for c in Commun[(y,measure)][index]:
        VoteshareAverage += Voteshares[y][c]
    VoteshareAverage = VoteshareAverage/len(Commun[(y,measure)][index])
    return VoteshareAverage



def NMI(confMat):
    N = np.sum(confMat)
    #print confMat
    Ni = np.sum(confMat,axis=1)
    Nj = np.sum(confMat,axis=0)
    #print Ni
    #print Nj
    
    Num = 0
    Denom1 = 0
    Denom2 = 0
    
    for i in range(len(Ni)):
        Denom1 += Ni[i]*np.log(Ni[i]/float(N))
        
    for j in range(len(Nj)):
        Denom2 += Nj[j]*np.log(Nj[j]/float(N))
    
    for i in range(len(Ni)):
        #Denom1 += Ni[i]*np.log(Ni[i]/float(N))
        for j in range(len(Nj)):
            #Denom2 += Nj[j]*np.log(Nj[j]/float(N))
            if confMat[i,j]!= 0:
                Num += confMat[i,j]*np.log(N*confMat[i,j]/float(Ni[i]*Nj[j]))
    return -2*Num/(Denom1 + Denom2)

def communMun2vec(C):
    y = np.zeros(290,int)
    count = 1
    for c in C:
        for m in c:
            y[munToNum['2018'][m]] = count
        count += 1
    return y

def sameColor(y,i,j):
    return int(y[i] == y[j])

################################################
#
#           Loading commuting data
#
################################################
print('Loading commuting data')

CommutingFile = 'Commuting_Sweden_1985-1998.xlsx'

xl = pd.ExcelFile(CommutingFile)



commuting = {}
commutingLabels = {}
Com = {}

years = xl.sheet_names[:-2]

for y in years:
    commuting[y]= xl.parse(y)
    if y == '1985':
        commutingLabels[y] = list(commuting[y].columns[1:])
        Com[y] = np.array(commuting[y].fillna(0).values[:,1:],int)
        
    else:
        commutingLabels[y] = list(commuting[y].columns[1:])
        Com[y] = np.array(commuting[y].fillna(0).values[:,1:],int)
        

#file = 'Commuting_Sweden_1985-1998.xlsx'
#xl = pd.ExcelFile(file)
#years = xl.sheet_names[:-2]

        
################################################
#
#           Loading Electoral data
#
################################################
print('Loading election data')

fileElec = 'Election_Sweden_1985-2018.xlsx'
xlElec = pd.ExcelFile(fileElec)
yearsElec = xlElec.sheet_names

dfelec = {}
labelselec = {}
municipalities = {}
Voteshares = {}


for y in yearsElec:
     dfelec[y]= xlElec.parse(y)
     labelselec[y] = list(dfelec[y].columns[1:])
     municipalities[y] = list(dfelec[y].values[:,0])#list(dfelec[y].index)
     Voteshares[y] = []
     for i in range(len(municipalities[y])):
         Voteshares[y].append(np.array(dfelec[y].fillna(0).values[i,1:],float)/np.sum(np.array(dfelec[y].fillna(0).values[i,1:],float)))


################################################
#
#           Loading Contiguity data
#
################################################
#print('Loading contiguity data')
#
fileNgh = 'Neighbors.xlsx'
xlNgh = pd.ExcelFile(fileNgh)
namesNgh = xlNgh.sheet_names
dfNgh = xlNgh.parse(namesNgh[0])
w = np.array(dfNgh.fillna(0).values[:,1:],int)
#
adjG = nx.from_numpy_matrix(w)
w = pysal.lib.weights.Rook.from_networkx(adjG)
#wShp = pysal.lib.weights.Rook.from_shapefile("../sweden1.shp")
################################################
#
#           Dictionaries to switch between indexing
#
################################################  
print('Computing dictionaries to switch between indexing')
    
nametonum = xl.parse(xl.sheet_names[-2])

values = np.array(nametonum.values[:,1],int)
numToName = dict(zip(values,nametonum.values[:,0]))
nameToNum = dict(zip(nametonum.values[:,0],values))

numtoRand = dict(zip(values,np.random.rand(290)))

numToMun = {}
munToNum = {}
for y in yearsElec:
    numToMun[y] = dict(zip(range(len(municipalities[y])),municipalities[y]))
    munToNum[y] = dict(zip(municipalities[y],range(len(municipalities[y]))))

################################################
#
#           Similarity
#
################################################
print('Computing similarity')
    
measures = ['Bhatt','Eucl','Cosine','JS']  

Dist = {}

 
#DistEucl = {}
#DistBhatt = {}
#DistCosine = {}
#DistJS = {}

for y in yearsElec:
    for m in measures:
        Dist[(y,m)] = np.zeros((len(municipalities[y]),len(municipalities[y])))
#    DistEucl[y] = np.zeros((len(municipalities[y]),len(municipalities[y])))
#    DistBhatt[y] = np.zeros((len(municipalities[y]),len(municipalities[y])))
#    DistCosine[y] = np.zeros((len(municipalities[y]),len(municipalities[y])))
#    DistJS[y] = np.zeros((len(municipalities[y]),len(municipalities[y])))

        for i in range(len(municipalities[y])):
            for j in range(len(municipalities[y])):
                if m == 'Bhatt':

                    Dist[(y,m)][i,j] =  bhattacharyya(Voteshares[y][i], Voteshares[y][j])

                elif m == 'Eucl':
                    Dist[(y,m)][i,j] = np.linalg.norm(Voteshares[y][i]-Voteshares[y][j])
                elif m == 'Cosine':
                    Dist[(y,m)][i,j] = max(spatial.distance.cosine(Voteshares[y][i], Voteshares[y][j]),0.)
                elif m == 'JS':

                    Dist[(y,m)][i,j] = js(Voteshares[y][i],Voteshares[y][j])
            #print str(i)+' '+str(j)
#            DistCosine[y][i,j] = max(spatial.distance.cosine(Voteshares[y][i], Voteshares[y][j]),0.)
#            DistBhatt[y][i,j] =  bhattacharyya(Voteshares[y][i], Voteshares[y][j])
#            DistEucl[y][i,j] = np.linalg.norm(Voteshares[y][i]-Voteshares[y][j])
#            DistJS[y][i,j] = js(Voteshares[y][i],Voteshares[y][j])
#    np.savetxt('Cosine'+y+'FullSweden.txt',DistCosine[y])
#    np.savetxt('Bhatt'+y+'FullSweden.txt',DistBhatt[y])
#    np.savetxt('Eucl'+y+'FullSweden.txt',DistEucl[y])
#    np.savetxt('JS'+y+'FullSweden.txt',DistJS[y])
            

    
    
################################################
#
#           Similarity distribution
#
################################################
print('Displaying similarity distribution')
    
print('JS')
for y in yearsElec:
    l = np.size(Dist[(y,'JS')],axis=1)
    histo = []
    for i in range(l):
        for j in range(i+1,l):
            histo.append(1-Dist[(y,'JS')][i,j])
    
    plt.hist(histo, density=True, histtype='step',bins=100,label=y) 
plt.legend()
plt.xlabel('JS')
plt.ylabel('Percent')
plt.savefig('SimDistrJS.png')
plt.show()

print('Bhatt')
meanBhatt = []
stdBhatt = []
for y in yearsElec:
    print(y)
    l = np.size(Dist[(y,'Bhatt')],axis=1)
    histo = []
    for i in range(l):
        for j in range(i+1,l):
            histo.append(1-Dist[(y,'Bhatt')][i,j])
    print(np.mean(histo))
    meanBhatt.append(np.mean(histo))
    stdBhatt.append(np.std(histo))
    print(np.std(histo))
    print(np.min(histo))
    pll = plt.hist(histo, density=True, histtype='step',bins=100,label=y) 
plt.legend()
plt.xlabel('BC')
plt.ylabel('Percent')
plt.savefig('SimDistrBC.eps')
plt.show()
print('Cosine')
for y in yearsElec:
    l = np.size(Dist[(y,'Cosine')],axis=1)
    histo = []
    for i in range(l):
        for j in range(i+1,l):
            histo.append(1-Dist[(y,'Cosine')][i,j])
    plt.hist(histo, density=True, histtype='step',bins=100,label=y) 
plt.legend()
plt.xlabel('Cosine similarity')
plt.ylabel('Percent')
plt.savefig('SimDistrCosine.png')
plt.show()
print('Euclide')
for y in yearsElec:
    l = np.size(Dist[(y,'Cosine')],axis=1)
    histo = []
    for i in range(l):
        for j in range(i+1,l):
            histo.append(1-Dist[(y,'Cosine')][i,j])
    plt.hist(histo, density=True, histtype='step',bins=100,label=y) 
plt.legend()
plt.xlabel('Euclidean similarity')
plt.ylabel('Percent')
plt.savefig('SimDistrEucl.png')
plt.show()


################################################
#
#           Community detection
#
################################################
print('Community detection...')

Commun = {}
dataToPlot = {}

for y in yearsElec:
    for m in measures:
        
        Commun[(y,m)] = []
        
        
        Ones = np.ones(np.shape(Dist[(y,m)]))
#        print(np.shape(Dist[(y,m)]))
        GG = nx.from_numpy_matrix(Ones - Dist[(y,m)])
        partition = community.best_partition(GG)
        size = float(len(set(partition.values())))
    
        count = 0.

#    f = open(outputfile,'w')
        for com in set(partition.values()) :
            count = count + 1.
            list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]

            Commun[(y,m)].append(list_nodes)
        
        Commun[(y,m)].sort(key=len)
        Commun[(y,m)].reverse()   
            

m='Bhatt'
Commun[('1985',m)][0], Commun[('1985',m)][1] = Commun[('1985',m)][1], Commun[('1985',m)][0]
for y in ['1985', '1988', '1991', '1994',]:
    Commun[(y,m)][2], Commun[(y,m)][1] = Commun[(y,m)][1], Commun[(y,m)][2]
    
Commun[('2018',m)][2], Commun[('2018',m)][3] = Commun[('2018',m)][3], Commun[('2018',m)][2]

m='JS'
Commun[('1985',m)][0], Commun[('1985',m)][1] = Commun[('1985',m)][1], Commun[('1985',m)][0]
for y in ['1985', '1988', '1991', '1994',]:
    Commun[(y,m)][2], Commun[(y,m)][1] = Commun[(y,m)][1], Commun[(y,m)][2]
    
Commun[('2018',m)][2], Commun[('2018',m)][3] = Commun[('2018',m)][3], Commun[('2018',m)][2]
    

m='Cosine'
Commun[('1985',m)][1], Commun[('1985',m)][2] = Commun[('1985',m)][2], Commun[('1985',m)][1]
Commun[('1988',m)][1], Commun[('1988',m)][2] = Commun[('1988',m)][2], Commun[('1988',m)][1]
#for y in ['1985', '1988', '1991', '1994',]:
#    Commun[(y,m)][2], Commun[(y,m)][1] = Commun[(y,m)][1], Commun[(y,m)][2]
#    
#Commun[('2018',m)][2], Commun[('2018',m)][3] = Commun[('2018',m)][3], Commun[('2018',m)][2]
CommunMun = {}
for y in yearsElec:
    for m in measures:
        CommunMun[(y,m)] = []
        for c in Commun[(y,m)]:
            com = []
            for el in c:
                #print el
                com.append(municipalities[y][el])
                if int(y) <= 1998:
                    if municipalities[y][el] == numToName[31]:
                        com.append(numToName[29])
                    # Knivsta in Uppsala 2003
                if int(y) <= 1994:
                    if municipalities[y][el] == numToName[18]:
                        com.append(numToName[13])
                    # Nykvarn part of Sodertalje 1999
                #    dataToPlot[(y,m)][13] = dataToPlot[(y,m)][18]
                if int(y) <= 1991:
                    if municipalities[y][el] == numToName[173]:
                        com.append(numToName[148])
                    if municipalities[y][el] == numToName[205]:
                        com.append(numToName[199])
                    # Bollebygds part of Borås 1995
                #    dataToPlot[(y,m)][148] = dataToPlot[(y,m)][173]
                    # Lekebergs part of Örebro 1995
                #    dataToPlot[(y,m)][199] = dataToPlot[(y,m)][205]
                if int(y) <= 1988:
                    if municipalities[y][el] == numToName[36]:
                        com.append(numToName[35])
                        com.append(numToName[42])
                # Gnesta and Trosa part of Nyköping 1992
                #dataToPlot[(y,m)][35] = dataToPlot[(y,m)][36]
                #dataToPlot[(y,m)][42] = dataToPlot[(y,m)][36]
            CommunMun[(y,m)].append(com)
            

############################################
#
#       Figure for identifying longitudinal communities
#
############################################  
northI = []
urbanI = []
rsouthI = []
fsouthI = []

yy = np.array(yearsElec,int)
objects = ('1985-1988', '1988-1991', '1991-1994', '1994-1998', '1998-2002', '2002-2006','2006-2010','2010-2014','2014-2018')
y_pos = yy[1:]

lss = ['o-','s-','d-.','v--']
cmap = plt.get_cmap('Accent')
norm = Normalize(vmin=1., vmax=5.)        
colors = [cmap(norm(1)),cmap(norm(2)),cmap(norm(3)),cmap(norm(4)),cmap(norm(1.5)),cmap(norm(5))]

print('Confusion matrix')
m = 'Bhatt'
for i in range(len(yearsElec)-1):
    m4 = confusionMatrix4(CommunMun[(yearsElec[i],m)],CommunMun[(yearsElec[i+1],m)])   
    print(m4[0:5,0:5])
    print(np.argmax(m4[:,0]))
    print(np.argmax(m4[:,1]))
    print(np.argmax(m4[:,2]))
    print(np.argmax(m4[:,3]))
    print(np.argmax(m4[:,4]))
    northI.append(m4[0,np.argmax(m4[0,0:3])])
    urbanI.append(m4[1,np.argmax(m4[1,0:3])]) 
    rsouthI.append(m4[2,np.argmax(m4[2,0:3])]) 
    fsouthI.append(m4[3,np.argmax(m4[3,0:3])])      

fsouthI[1]=0.

plt.plot(yy[1:],northI,lss[0],label='North',color=cmap(norm(1)))
plt.plot(yy[1:],urbanI,lss[1],label='Urban',color=cmap(norm(2)))
plt.plot(yy[1:],rsouthI,lss[2],label='R. South',color=cmap(norm(3)))
plt.plot(yy[-4:],fsouthI[-4:],lss[3],label='F. South',color=cmap(norm(4)))
np.set_printoptions(precision=3)    

plt.legend()
plt.ylabel('Index I')
plt.xticks(y_pos,objects)
plt.xticks(rotation=15)
plt.savefig('IndexLongCom.eps',format='eps')
plt.show()    
#          
############################################
#
#       Identification of communities from year to year
#
############################################  
            
#print(JaccardMatrix(Commun[('1991','Bhatt')],Commun[('1988','Bhatt')]))
#            
##orderCom = {}
##            
##for y in yearsElec:
##    for m in measures:
##        pass
#            
#            
#        dataToPlot[(y,m)] = {}
for y in yearsElec:
    for m in ['Bhatt']:
        dataToPlot[(y,m)] = {}
        #count = 0
        for i in range(len(Commun[(y,m)])):
            #count += 1
            #print y,m,count
            for c in Commun[(y,m)][i]:
                if len(Commun[(y,m)][i]) >= 8:
                    dataToPlot[(y,m)][nameToNum[numToMun[y][c]]] = i+1
                else:
                    dataToPlot[(y,m)][nameToNum[numToMun[y][c]]] = 6#count + 100
                
        if int(y) <= 1998:
            # Knivsta in Uppsala 2003
            dataToPlot[(y,m)][29] = dataToPlot[(y,m)][31]
        if int(y) <= 1994:
            # Nykvarn part of Sodertalje 1999
            dataToPlot[(y,m)][13] = dataToPlot[(y,m)][18]
        if int(y) <= 1991:
            # Bollebygds part of Borås 1995
            dataToPlot[(y,m)][148] = dataToPlot[(y,m)][173]
            # Lekebergs part of Örebro 1995
            dataToPlot[(y,m)][199] = dataToPlot[(y,m)][205]
        if int(y) <= 1988:
            # Gnesta and Trosa part of Nyköping 1992
            dataToPlot[(y,m)][35] = dataToPlot[(y,m)][36]
            dataToPlot[(y,m)][42] = dataToPlot[(y,m)][36]
    


#
##
##


for y in yearsElec:
    for m in ['Bhatt']:


        # Create figure
        fig, ax = plt.subplots(figsize=(5,10))

        # Load the map in the region of Sweden

        ma = Basemap(llcrnrlon=11.,llcrnrlat=55.,urcrnrlon=26,urcrnrlat=69.3,
            resolution='c',projection='cass',lon_0=17.5,lat_0=62)
    

        # Get the boundaries of Sweden municipalities from a shapefile
        ma.readshapefile("../sweden1","sweden", drawbounds = True)
        patches   = []
        facec = []


        

        for info, shape in zip(ma.sweden_info, ma.sweden):
            patches.append( Polygon(np.array(shape), True) )
            #facec.append(color_list[min(dataToPlot[(y,m)][info['SHAPENUM']]-1,6)])
            facec.append(colors[dataToPlot[(y,m)][info['SHAPENUM']]-1])
                   
                    
        # Create patch collection  
        pc = PatchCollection(patches, edgecolor='k', linewidths=1., zorder=2)
        norm = Normalize(vmin=1., vmax=5.)

        # Initialize colormap
        cmap = plt.get_cmap('Accent')   

        # Assign color to faces
        #pc.set_facecolor(facec)#cmap(norm(facec)))
        pc.set_facecolor(facec)#cmap(norm(facec)))
        ax.add_collection(pc)

        # Create color bar
        #mapper = matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap)# 
 
        #mapper.set_array(facec)
        #plt.colorbar(mapper, shrink=0.4)
        plt.tight_layout()
        #plt.title(y,fontsize=20)
        plt.annotate(y, xy=(0.1, 0.8), xycoords='axes fraction',fontsize = 50)
        plt.savefig('Communities'+m+y+'.eps',format='eps')
        #plt.savefig('Communities'+m+y+'.png')
        print(m)
        plt.show()


            
############################################
#
#       Joint count statistics for k-colored factors
#
############################################  
print('Joint count statistics for k-colored factors')
m = 'Bhatt'
for year in yearsElec:
    print(year)
    y = communMun2vec(CommunMun[(year,'Bhatt')])
    g = Gamma(y,w,operation = sameColor)
    print('Gamma value: ' + str(g.g))
    print('Max Gamma: ' + str(g.max_g))
    print('z values: ' + str(g.g_z))
    print('Pseudo p-value: ' + str(g.p_sim_g))
            


#C = confusionMatrix(Commun[('2018','Bhatt')],Commun[('2018','JS')])
##
##
##
##distBhattJS = []
##distBhattCos = []
##distBhattEucl = []
##distEuclJS = []
##distEuclCos = []
##distCosJS = []
##
##for y in yearsElec:
##    distBhattJS.append(mapDistance(dataToPlot[(y,'Bhatt')],dataToPlot[(y,'JS')]))
##    distBhattCos.append(mapDistance(dataToPlot[(y,'Bhatt')],dataToPlot[(y,'Cosine')]))
##    distBhattEucl.append(mapDistance(dataToPlot[(y,'Bhatt')],dataToPlot[(y,'Eucl')]))
##    distEuclJS.append(mapDistance(dataToPlot[(y,'Eucl')],dataToPlot[(y,'JS')]))
##    distEuclCos.append(mapDistance(dataToPlot[(y,'Eucl')],dataToPlot[(y,'Cosine')]))
##    distCosJS.append(mapDistance(dataToPlot[(y,'Cosine')],dataToPlot[(y,'JS')]))
##    

##plt.plot(yy,distBhattJS,'k')
##plt.plot(yy,distBhattCos,'r:')
###plt.plot(yy,distBhattEucl,'r--')
###plt.plot(yy,distEuclJS,'g--')
###plt.plot(yy,distEuclCos,'k')
##plt.plot(yy,distCosJS,'g:')
##plt.xlabel('years')
##plt.ylabel('distance')
##plt.title('Normalized Hamming distance between maps')
##plt.savefig('MapHammingDist.png')
##
##evBhatt = []
##evJS = []
##evEucl = []
##evCos = []
##
##
##for i in range(len(yearsElec[:-1])):
##    evBhatt.append(mapDistance(dataToPlot[(yearsElec[i],'Bhatt')],dataToPlot[(yearsElec[i+1],'Bhatt')]))
##    evJS.append(mapDistance(dataToPlot[(yearsElec[i],'JS')],dataToPlot[(yearsElec[i+1],'JS')]))
##    evEucl.append(mapDistance(dataToPlot[(yearsElec[i],'Eucl')],dataToPlot[(yearsElec[i+1],'Eucl')]))
##    evCos.append(mapDistance(dataToPlot[(yearsElec[i],'Cosine')],dataToPlot[(yearsElec[i+1],'Cosine')]))
##    
##plt.plot(yy[1:],evBhatt)
##plt.plot(yy[1:],evJS)
##plt.plot(yy[1:],evEucl)
##plt.plot(yy[1:],evCos)
##plt.show()
##
lss = ['o-','s-','d-.','v--']
leg = ['North','Urban','R South','F South','Rest']
for i in range(4):
    sizeBhatt = []
    for y in yearsElec:
    
        sizeBhatt.append(len(CommunMun[(y,'Bhatt')][i]))
    if i <= 2:
        plt.plot(yy,sizeBhatt,lss[i],color=cmap(norm(i+1)),linewidth=1,label=leg[i])
        print(i)
        print(sizeBhatt)
    else:
        plt.plot(yy[-5:],sizeBhatt[-5:],'v--',color=cmap(norm(i+1)),linewidth=1,label=leg[i])
        correction = sizeBhatt[0:-5]+5*[0]
plt.plot(2018,len(CommunMun[(y,'Bhatt')][4]),'*', color = colors[4], label='New Com')
        #print(i)
        #print(sizeBhatt)
    #print cmap(norm(i+1))
rest = []
for y in yearsElec:
    rest.append(290-len(CommunMun[(y,'Bhatt')][0])-len(CommunMun[(y,'Bhatt')][1])-len(CommunMun[(y,'Bhatt')][2])-len(CommunMun[(y,'Bhatt')][3]))
rest[-1] -= 10
plt.plot(yy,np.array(rest)+np.array(correction),'k:')
print(np.array(rest)+np.array(correction))
plt.xticks(yy, yy)
plt.legend()
plt.xlabel('Years')
plt.ylabel('# Municipalities')
#plt.title('Kluster storlek')
plt.savefig('ComSizeEvo2.eps',format='eps')
plt.show()

numCom = []
for y in yearsElec:
    numCom.append(len(Commun[(y,'Bhatt')]))
    
plt.plot(yy,numCom,'k')
print(numCom)
plt.xticks(yy, yy)
plt.xlabel('years')
plt.ylabel('Number of communities')
#plt.title('Number of communities')
plt.savefig('NumberCommunities.eps',format='eps')
plt.show()
##

##
#Voteshares['1985'][slice(np.array(Commun[('1985','Bhatt')][0],dtype=int))]
communityMatrice = {}
for y in yearsElec:
    for i in range(4):
        communityMatrice[(y,i)] = []
        for c in Commun[(y,'Bhatt')][i]:
            communityMatrice[(y,i)].append(list(Voteshares[y][c]))

for i in range(4):
    print(i)
    for y in yearsElec:
        #print(y)
        print(np.std(communityMatrice[(y,i)],axis=0)/np.mean(communityMatrice[(y,i)],axis=0))
        
communityProt = {}
for i in range(4):
    for y in yearsElec:
        #print(y)
        communityProt[(y,i)] = np.mean(communityMatrice[(y,i)],axis=0)
        #print(np.mean(communityMatrice[(y,i)],axis=0))
        
BCtocenter = {}
meanBCtocenter = {}
stdBCtocenter = {}
ncom = 4
for i in range(ncom):
    for y in yearsElec:
        BCtocenter[(y,i)] = []
        
        for c in communityMatrice[(y,i)]:
            BCtocenter[(y,i)].append(1.-bhattacharyya(c,communityProt[(y,i)]))
        meanBCtocenter[(y,i)] = np.mean(np.array(BCtocenter[(y,i)]))
        stdBCtocenter[(y,i)] = np.std(np.array(BCtocenter[(y,i)]))
        
for i in range(ncom):
    #print(i)
    plotstd = []
    for y in yearsElec:
        #print(meanBCtocenter[(y,i)])
        plotstd.append(stdBCtocenter[(y,i)])
    if i == 3:
        
        plt.plot(yy[5:],plotstd[5:],color=cmap(norm(i+1)),linewidth=2)
    else:
        plt.plot(yy,plotstd,color=cmap(norm(i+1)),linewidth=2)
    plt.xticks(yy, yy)
plt.xlabel('year')
plt.ylabel('Std of BC to community prototype')
plt.savefig('StdBCtoPrototype.png')
plt.show()
##
lss = ['o-','s-','d-.']
for i in range(ncom):
    #print(i)
    plotmean = []
    plotstd = []
    for y in yearsElec:
        #print(meanBCtocenter[(y,i)])
        
        plotmean.append(meanBCtocenter[(y,i)])
        plotstd.append(stdBCtocenter[(y,i)])
    if i == 3:
        plt.errorbar(yy[5:],plotmean[5:],yerr = plotstd[5:],color=cmap(norm(i+1)),linewidth=1,capsize=10,fmt='v--',alpha=0.7)
        #plt.plot(yy[5:],plotmean[5:],color=cmap(norm(i+1)),linewidth=2,'--')
    else:
        plt.errorbar(yy,plotmean,yerr = plotstd,color=cmap(norm(i+1)),linewidth=1,capsize=10,fmt=lss[i],alpha=0.7)
        #plt.plot(yy,plotmean,color=cmap(norm(i+1)),linewidth=2)
    plt.xticks(yy, yy)
plt.xlabel('Years')
plt.ylabel('Mean BC to community prototype')
plt.savefig('MeanBCtoPrototype2.eps',format='eps')
plt.show()
##
##
##
for i in range(ncom):
    #print(i)
    plotCV = []
    for y in yearsElec:
        #print(meanBCtocenter[(y,i)])
        plotCV.append(stdBCtocenter[(y,i)]/meanBCtocenter[(y,i)])
    if i == 3:
        plt.plot(yy[5:],plotCV[5:],color=cmap(norm(i+1)),linewidth=2)
    else:
        plt.plot(yy,plotCV,color=cmap(norm(i+1)),linewidth=2)
    #plt.plot(yy,plotCV,color=cmap(norm(i+1)),linewidth=2)
    plt.xticks(yy, yy)
plt.xlabel('year')
plt.ylabel('CV of BC to community prototype')
plt.savefig('CVBCtoPrototype.png')
plt.show()
##
prototypeVS = {}
for y in yearsElec:
    for i in range(5):
        prototypeVS[(y,i)] = AverageVoteShareInCom(Commun,y,'Bhatt',Voteshares,i)
        
nnn = len(Commun[('1985','Bhatt')])
fulll = []
for i in range(nnn):
    fulll += list(Commun[('1985','Bhatt')][i])


#np.set_printoptions(precision=2)
#for y in yearsElec:
#    print(100*np.mean(Voteshares[y], axis = 0))
##np.mean(Voteshares['1985'], axis = 0)
#
#np.set_printoptions(precision=2)
#for y in yearsElec:
#    print(np.std(Voteshares[y], axis = 0)/np.mean(Voteshares[y], axis = 0))
    
objects = ('M', 'C', 'L', 'KD', 'S', 'V','MP','SD','Other','Invalid','Non Voters')
y_pos = np.arange(len(objects))
for y in yearsElec:
    for i in range(4):
        performance = prototypeVS[(y,i)]
        print(y)
        print(i)
        print(100*performance)
        print(np.around((performance - np.mean(Voteshares[y], axis = 0))/np.std(Voteshares[y], axis = 0),decimals=2))

print('Fifth com')
performance =   prototypeVS[('2018',4)]  
print(np.around((performance - np.mean(Voteshares['2018'], axis = 0))/np.std(Voteshares['2018'], axis = 0),decimals=2))
   
objects = ('M', 'C', 'L', 'KD', 'S', 'V','MP','SD','Other','Invalid','Non Voters')
y_pos = np.arange(len(objects))

performance = prototypeVS[('1985',0)]
plt.xticks(rotation=15)
plt.bar(y_pos, 0*performance, align='center',color=['dodgerblue','forestgreen','royalblue','mediumblue','r','firebrick','yellowgreen','gold','grey','lightgrey','k'])
plt.xticks(y_pos, objects)
plt.ylabel('Vote share')
plt.title('Vote share in community '+str(i))
plt.tight_layout()
plt.ylim((0, 0.53))
plt.savefig('PrototypeEmpty.png')
plt.show()
##
##
##
comNames = ['North','Urban','R. South','F. South']
colors = ['ob--','sr--','dg--','vy-.','^m-.','pc:']
res = {}
res2 = {}
k=0
for i in range(3):
    for j in range(i+1,4):
#        print(k)
        res[(i,j)] = []
        res2[(i,j)] = []
        for y in yearsElec:
            res[(i,j)].append(abs(1-bhattacharyya(prototypeVS[(y,i)], prototypeVS[(y,j)])))
            res2[(i,j)].append(np.exp(-(bhattacharyya(prototypeVS[(y,i)], prototypeVS[(y,j)]))))
        #plt.plot(yy,res[(i,j)],label=comNames[i]+' - '+comNames[j])
        if j <= 2:
            plt.plot(yy,res[(i,j)],colors[k],label=comNames[i]+' - '+comNames[j],lw=1)
        else:
            plt.plot(yy[-5:],res[(i,j)][-5:],colors[k],label=comNames[i]+' - '+comNames[j],lw=1)    
        k+=1
plt.plot(yy,meanBhatt,'k',label='mean BC')
plt.xticks(yy, yy)
plt.xlabel('Election year')
plt.ylabel('BC')
#plt.legend(bbox_to_anchor=(1.04,1), loc="upper left", fontsize=12)
plt.legend()
plt.savefig('ComDist2.eps',format='eps', bbox_inches="tight")
plt.show()
##
##
##
nmiBJ = []
nmiBC = []
nmiJC = []
nmiCE = []
for y in yearsElec:
    nmiBJ.append(NMI(confusionMatrix(CommunMun[(y,'Bhatt')],CommunMun[(y,'JS')])))
    nmiBC.append(NMI(confusionMatrix(CommunMun[(y,'Bhatt')],CommunMun[(y,'Cosine')])))
    nmiJC.append(NMI(confusionMatrix(CommunMun[(y,'JS')],CommunMun[(y,'Cosine')])))
plt.plot(yy,nmiBJ,'k',label='Bhatt - JS')
plt.plot(yy,nmiBC,'r:',label='Bhatt - Cosine')
plt.plot(yy,nmiJC,'g:',label='JS - Cosine')
#plt.plot(yy,nmiCE,'m--',label='Cosine - Euclidean')
plt.xticks(yy, yy)
plt.legend()
plt.xlabel('years')
plt.ylabel('NMI')
#plt.plot(yy,nmiCE)
plt.axis([1985,2018,0.4,1])
plt.savefig('NMINew.eps')
plt.show()
##
###CC = confusionMatrix(Commun[(y,'Bhatt')],Commun[(y,'Eucl')])
##
evnmiBhatt = []
evnmiJS = []
evnmiEucl = []
evnmiCos = []

objects = ('1985-1988', '1988-1991', '1991-1994', '1994-1998', '1998-2002', '2002-2006','2006-2010','2010-2014','2014-2018')
y_pos = yy[1:]


for i in range(len(yearsElec[:-1])):
    evnmiBhatt.append(NMI(confusionMatrix(CommunMun[(yearsElec[i],'Bhatt')],CommunMun[(yearsElec[i+1],'Bhatt')])))
    evnmiJS.append(NMI(confusionMatrix(CommunMun[(yearsElec[i],'JS')],CommunMun[(yearsElec[i+1],'JS')])))
    evnmiEucl.append(NMI(confusionMatrix(CommunMun[(yearsElec[i],'Eucl')],CommunMun[(yearsElec[i+1],'Eucl')])))
    evnmiCos.append(NMI(confusionMatrix(CommunMun[(yearsElec[i],'Cosine')],CommunMun[(yearsElec[i+1],'Cosine')])))
    
plt.plot(yy[1:],evnmiBhatt,label='Bhatt')
np.set_printoptions(precision=3)
print(np.array(evnmiBhatt))
#plt.plot(yy[1:],evnmiJS,label='JS')
#plt.plot(yy[1:],evnmiEucl,label='Eucl')
#plt.plot(yy[1:],evnmiCos,label = 'Cosine')

#plt.legend()
plt.ylabel('Normalized Mutual Information')
plt.xticks(y_pos,objects)
plt.xticks(rotation=15)
plt.savefig('NMIPredict.eps',format='eps')
plt.show()
##
##
