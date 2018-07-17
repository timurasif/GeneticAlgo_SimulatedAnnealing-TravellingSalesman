import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as fig
from matplotlib import style
import csv
import string
from collections import Counter
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
import collections, re
import random
from difflib import SequenceMatcher
import pandas as pd
from math import sin, cos, sqrt, atan2, radians
from numpy import loadtxt
from random import randint
from boltons import iterutils
import copy
from tqdm import tqdm
from random import shuffle



def make_chromosome(chromosomes):
    "Makes a chromosome which is not present in chromosomes"
    chrm=[]
    i=0
    for i in range(312):
        index=randint(0, 311)
        while index in chrm:
            index=randint(0, 311)
        chrm.append(index)
    if chrm not in chromosomes:
        return chrm
    else:
        return False

def fitness_function(chromosome):
    "Returns integer value for the fitness function, the larger the better"
    #print("in fittt")
    fitness=0
    i=1
    d=distances[chromosome[0]][chromosome[1]]
    j=1
    for i in range(len(chromosome)-2):
        j+=1
        d=d+distances[chromosome[i]][chromosome[i+1]] 
    i-=1
    d=d+distances[chromosome[i]][chromosome[0]]     #coming to original point
    fitness=d/100000.0     #making it in range of 3-4 
    fitness=5-fitness       #conversing the fitness
    #print(fitness)
    return fitness

def crossover(chromosome1,chromosome2):
    "Does the crossover i.e. random selection from both parents"
    offspring=[]
    j=0
    i=0
    index1=0
    for i in range(156):
        if(j>311):
            break
        index1=randint(0, 311)
        element1=chromosome1[index1]
        while element1 in offspring:
            index1=randint(0, 311)
            element1=chromosome1[index1]
        offspring.append(element1)
        j+=1

        index2=randint(0, 311)
        element2=chromosome2[index2]
        while element2 in offspring:
            index2=randint(0, 311)
            element2=chromosome2[index2]
        offspring.append(element2)
        j+=1

    return offspring   

def fillchild(chm,t):
    "Fills values of given chromosome in a child"
    i=0
    for i in range(len(chm)-1):
        if chm[i] not in t:
            #print("appending ")
            t.append(chm[i])



    

def myOwnCrossover(chm1,chm2):
    "Extracts minimum distance of each half from each parent"
    a1=copy.deepcopy(chm1)
    a11=copy.deepcopy(chm1)
    a2=copy.deepcopy(chm2)
    a22=copy.deepcopy(chm2)
    r1=a1[0:155]
    r11=a11[156:311]
    r2=a2[0:155]
    r22=a22[156:311]
    f1=fitness_function(r1)
    f11=fitness_function(r11)
    f2=fitness_function(r2)
    f22=fitness_function(r22)
    t=[]
    if f1>=f11 and f1>=f2 and f1>=f22:
        fillchild(r1,t)
        fillchild(chm2,t)
    elif f11>=f1 and f11>=f2 and f11>=f2:
        fillchild(r11,t)
        fillchild(chm2,t)
    if f2>=f11 and f2>=f1 and f2>=f22:
        fillchild(r2,t)
        fillchild(chm1,t)
    if f22>=f11 and f22>=f2 and f22>=f1:
        fillchild(r22,t)
        fillchild(chm1,t)

    ''' print("f1: ",f1)
    print("f11: ",f11)
    print("f2: ",f2)
    print("f22: ",f22)
    '''
    ''' print("chm1: ",chm1)
    print("fit1: ",fitness_function(chm1))
    print("chm2: ",chm2)
    print("fit2: ",fitness_function(chm2))
    print("T: ",t)
    print("fit T: ",fitness_function(t))
    print("LEN: ",len(t))
     '''
    return t
    


def Tournament_Selection(parents,K):
    "Does TS to pick one parent, have K parents and select one from them"
    if K<=2:
        index1=randint(0,1)
        return parents[index1]
    elif K==3:
        index1=randint(0,2)
        return parents[index1]
    k=3
    index1=randint(0,K-1)
    index2=randint(0,K-1)
    index3=randint(0,K-1)
    while index1==index2 or index2==index3 or index1==index3:
        index2=randint(0,K-1)
        index3=randint(0,K-1)
    f1=fitness_function(parents[index1])
    f2=fitness_function(parents[index2])
    f3=fitness_function(parents[index3])
    if f1>f2 and f1>f3:
        return parents[index1]
    elif f2>f1 and f2>f3:
        return parents[index2]
    else:
        return parents[index3]

def myOwnSelectionMethod(parents,K):
    '''My own selection algo, shuffles list of parents then selects randomnly 3
    and then 1 by crossovering them'''
    shuffle(parents)
    if K>=4:
        index1=0
        index2=K-1
        index3=2
    else:
        return parents[0]
    i=0
    j=0
    offspring=[]
    chromosome1=parents[index1]
    chromosome2=parents[index2]
    chromosome3=parents[index3]
    for i in range(104):
        if(j>311):
            break
        index1=randint(0, 311)
        element1=chromosome1[index1]
        while element1 in offspring:
            index1=randint(0, 311)
            element1=chromosome1[index1]
        offspring.append(element1)
        j+=1

        index2=randint(0, 311)
        element2=chromosome2[index2]
        while element2 in offspring:
            index2=randint(0, 311)
            element2=chromosome2[index2]
        offspring.append(element2)
        j+=1

        index3=randint(0, 311)
        element3=chromosome3[index3]
        while element3 in offspring:
            index3=randint(0, 311)
            element3=chromosome3[index3]
        offspring.append(element3)
        j+=1
    offspring.append(offspring[0])

    return offspring


def Mutate(chromosome):
    "Does inversion mutation of the chromosome given"
    k1=4
    k2=11
    #chromosome[k1:k2] will be reversed now from 4 to 10
    chromosome[k1:k2] = reversed(chromosome[k1:k2])
    return chromosome

def Mutate2(chromosome):
    "Does Scramble Mutation here"
    k1=40
    k2=100
    shuffle(chromosome[k1:k2])
    return chromosome

def Mutate3(chromosome):
    "Swaps cities which have high distance with low ones"
    d1=distances[chromosome[0]][chromosome[1]]
    x=0
    y1=1
    y2=2
    j=0
    while j<300:
        d1=distances[chromosome[x]][chromosome[y1]]
        d2=d1+2
        y2=y1+1
        flag=0
        while d2>=d1:
            if y2>=312:
                flag=1
                break
            d2=distances[chromosome[x]][chromosome[y2]]
            y2+=1
        if flag==1:
            j+=1
            x+=1
            y1+=1
        else:
            y2-=1
            chromosome[y1],chromosome[y2]=chromosome[y2],chromosome[y1]
            j+=1
            x+=1
            y1+=1
    #print("FITNESS AFTER MUTATION: ",fitness_function(chromosome))
    return chromosome

def new_population(fit_chrms,K):
    "Gets best chromosomes from children and parents and returns new parents"
    fit=[]
    i=0
    max1=fit_chrms[0]        #getting 1st
    j=0
    for i in range(K):
        for j in range(len(fit_chrms)):
            if max1[1]<fit_chrms[j][1]:
                max1=copy.deepcopy(fit_chrms[j])
        fit.append(max1[0])
        fit_chrms.remove(max1)
        max1=[[0],0.01]

    return fit

def get_bestchrm(fit_chrms):
    "Returns best chromosome in the entire population"
    max1=fit_chrms[0]
    for m in range(len(fit_chrms)):
        if max1[1]<fit_chrms[j][1]:
            max1=copy.deepcopy(fit_chrms[j])
    return max1


def addfitness(fit_chrms,parents):
    "Adds fitness in parents and add them in fit chrms"
    for x in range(len(parents)):
        s=[]
        f=fitness_function(parents[x])
        s.append(parents[x])
        s.append(f)
        fit_chrms.append(s)

def total_distance(chromosome):
    "Tells total distance taken by the chromosome"
    d=distances[chromosome[0]][chromosome[1]]
    j=1
    for i in range(310):
        j+=1
        d=d+distances[chromosome[i]][chromosome[i+1]] 
    i-=1
    d=d+distances[chromosome[i]][chromosome[0]]     #coming to original point
    return d



def makeint(arr):
    v=str()
    if arr:
        for c in arr:
            v+=c
        return v  
    else:
        return 0

def converttoint(citydist):
    temp=[]
    for a in citydist:
        temp.append(int(a))
    return temp

def converttofloat(xy):
    i=0
    for i in range(len(xy)):
        xy[i]=float(xy[i])
    

#making dimensions here of latitude, longitude
f = open('usca312_dms.txt', 'r')
x=f.readline()
x=f.readline()
x=f.readline()
x=f.readline()
dimensions=[]
i=0
while x:
    i+=1
    x=f.readline()
    m=str(x)
    arr=[]
    dms=[]
    for s in m:
        if (s==" ") or (s=="\n") or (s=="N") or (s=="W"):
            r=makeint(arr)
            if r!=0:
                dms.append(r)
            arr=[]
            continue
        else:
            arr.append(s)
    l=0
    q=iterutils.chunked(dms, 3)     #chunking in parts of three
    if q:
        dimensions.append(q)
#print(dimensions)           #got dimensions here


#getting xy co-ordinates of each city 
i=0
f = open('usca312_xy.txt', 'r')
x=f.readline()
x=f.readline()
x=f.readline()
x=f.readline()
x=f.readline()
x=f.readline()
x=f.readline()
coordinates=[]
while x:
    i+=1
    x=f.readline()
    #print(x)
    m=str(x)
    arr=[]
    xy=[]
    for s in m:
        if (s==" ") or (s=="\n"):
            r=makeint(arr)
            #print("R: ",r)
            if r!=0:
                xy.append(r)
            arr=[]
            continue
        else:
            arr.append(s)   
    converttofloat(xy)
    if xy:
        coordinates.append(xy)
#print(coordinates)                  #got co-ordinates

#getting distances heres
f = open('usca312_dist.txt', 'r')
i=0
x = f.readline()
x=f.readline()
x=f.readline()
x=f.readline()
x=f.readline()
x=f.readline()
x=f.readline()
distances=[]
while x:
    citydist=[]
    x=0
    for m in range(32):
        x=f.readline()
        m=str(x)
        arr=[]
        j=0
        for s in m:
            j+=1
            #print("S: ",s)
            if (s==" ") or (s=="\n"):
                #print(arr)
                if arr:
                    r=makeint(arr)
                    citydist.append(r)
                    arr=[]
                continue
            else:
                arr.append(s)
    #print(citydist)
    sublist=converttoint(citydist)
    if sublist:
        distances.append(sublist)

#getting cities' name
f = open('usca312_name.txt', 'r')
x = f.readlines()
#print(x[15:])
cities=x[15:]         #got list of all city names

''' print("CITIES: ",cities)

print("Ditances between ",cities[0]," and ",cities[3], " is: ",
distances[0][3], " or is ", distances[3][0])
 '''
#Use tournament selection now

K = int(input("Enter size of population: "))
print("population: ",K)

while(K<=1):
    print("Please enter more than 1 parents")
    K = int(input("Enter size of population: "))
    print("population: ",K)
    

selection = int(input("Enter method of selection 1 for Tournament Selection"+
" and any other integer for My Own Selection Selection: "))
print("Selection: ",selection)

max_iterations = int(input("Enter max iterations: "))
print("Max Iterations: ",max_iterations)

mutation_choice = int(input("Enter 1 for Inversion Mutation, 2 " +
 "for Scarmble Mutation and 3 for my own mutation: "))
print("mutation_choice: ",mutation_choice)

cross=int(input("Enter 1 for 1 point cross over and any other integer for "+
"my own cross over:  "))
print("Crossover: ",cross)

fit_chrms=[]

#making chromosome
parents=[]
i=0
#making K parents initially
for i in range(K):
    sub=[]
    c1=make_chromosome(parents)
    while c1==False:
        print("Got same!!")
        c1=make_chromosome(parents)
    parents.append(c1)
    f=fitness_function(c1)
    sub.append(c1)
    sub.append(f)
    fit_chrms.append(sub)



#print(len(parents[0]))


i=0
for i in range(K):
    print("Fitness of parent ",i+1," is: ",fitness_function(parents[i]))

Mutate3(parents[2])

p=0
with tqdm(total=max_iterations) as pbar:
    while p<max_iterations:
        for o in range(len(parents)-1):
            subs=[]
            if selection ==1:
                p1=Tournament_Selection(parents,K)
                p2=Tournament_Selection(parents,K)
            else:
                p1=myOwnSelectionMethod(parents,K)
                p2=myOwnSelectionMethod(parents,K)
            while p1==p2:
                if selection ==1:
                    p2=Tournament_Selection(parents,K)
                else:
                    p2=myOwnSelectionMethod(parents,K)   
            if cross==1:
                l=crossover(p1,p2)
            else:
                l=myOwnCrossover(p1,p2)
            f=fitness_function(l)
            if f<1.3 or f<2 or f<7:       #Mutating only if it has low fitness
                if mutation_choice==1:
                    l=Mutate2(l)
                elif mutation_choice==2:
                    l=Mutate(l)
                else:
                    l=Mutate3(l)
                f=fitness_function(l)
            subs.append(l)
            subs.append(f)
            fit_chrms.append(subs)
        parents=new_population(fit_chrms,K)
        fit_chrms=[]
        addfitness(fit_chrms,parents)
        p+=1
        pbar.update(1)

print("Total iterations took: ",max_iterations)
best=get_bestchrm(fit_chrms)
print("BEST: ",best)
print("LEN: ",len(best[0]))
print("Fitness: ",best[1])


print("Total distance travelled to visit 312 cities: ",total_distance(best[0]), " km.")


'''Visualizing the cities now'''
 
x=[]
y=[]
for w in range(len(coordinates)):
    x.append(coordinates[w][0])
    y.append(coordinates[w][1])

xlab="x-axis"
ylab="y-axis"
title="Distance"

plt.xlabel(xlab)
plt.ylabel(ylab)
plt.title(title)

x=np.array(x)
y=np.array(y)

plt.scatter(x,y,c="red",s=2,alpha=.9)
plt.show()

w=0
for w in range(len(best[0])-1):
    plt.plot(x[w:w+2], y[w:w+2], 'ro-')

plt.show()