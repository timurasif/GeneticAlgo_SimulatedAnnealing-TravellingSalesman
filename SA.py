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



def make_chromosome():
    "Makes a chromosome which is not present in chromosomes"
    chrm=[]
    i=0
    for i in range(312):
        index=randint(0, 311)
        while index in chrm:
            index=randint(0, 311)
        chrm.append(index)
    return chrm

def successor_function(chromosome):
    "Successor function, reverses portion b/w two cities"
    k1=randint(0,311)
    k2=randint(0,311)
    chrm=copy.deepcopy(chromosome)
    chrm[k1],chrm[k2]=chrm[k2],chrm[k1]
    return chrm


def fitness_function(chromosome):
    "Returns integer value for the fitness function, the larger the better"
    fitness=0
    i=1
    d=distances[chromosome[0]][chromosome[1]]
    j=1
    for i in range(310):
        j+=1
        d=d+distances[chromosome[i]][chromosome[i+1]] 
    i-=1
    d=d+distances[chromosome[i]][chromosome[0]]     #coming to original point
    fitness=d/100000.0     #making it in range of 3-4 
    fitness=5-fitness       #conversing the fitness
    return fitness

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
    

def decrease_temperature(temperature,iterations):
    "Decreases the temperature by a specific ratio after 1 iteration"
    temperature=temperature-((iterations/max_iterations)*10)
    return temperature


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

temperature = int(input("Enter initial value of temperature: "))
print("Temperature :  ",temperature)

max_iterations = int(input("Enter max iterations: "))
print("Max_Iterations: ",max_iterations)


fit_chrms=[]

i=0
#making a random state initially
sub=[]
c1=make_chromosome()
while c1==False:
    print("Got same!!")
    c1=make_chromosome()
parents=c1
parents=Mutate3(parents)
sub.append(parents)
f=fitness_function(parents)
sub.append(f)
fit_chrms.append(sub)


i=0
print("Fitness is: ",fitness_function(parents))

iterations=0
#max_iterations=1000000
#temperature=100
probability=0

p1=0
with tqdm(total=max_iterations) as pbar:
    while p1<max_iterations:
        if temperature<=0:
            probability=0
        subs=[]
        p=successor_function(parents)
        new_temperature=decrease_temperature(temperature,iterations)
        if temperature != 0:
            probability=new_temperature/temperature*100
            temperature=new_temperature
        f1=fitness_function(parents)
        f2=fitness_function(p)
        if temperature<=0:
            temperature=0
            probability=0
        if probability>=90:
            if f2>f1:       #accepting on high prob
                parents=p
        elif probability>=80:       #accepting if new one is somehow good
            if f2-f1>=0.01:
                parents=p
        elif probability>=70:       #accepting if new one is somehow good
            if f2-f1>=0.03:
                parents=p
        elif probability>=60:       #accepting if new one is somehow good
            if f2-f1>=0.05:
                parents=p
        elif probability>=50:       #accepting if new one is somehow good
            if f2-f1>=0.07:
                parents=p
        elif probability>=40:       #accepting if new one is somehow good
            if f2-f1>=0.09:
                parents=p
        elif probability>=30:       #accepting if new one is somehow good
            if f2-f1>=0.10:
                parents=p
        elif probability>=20:       #accepting if new one is somehow good
            if f2-f1>=0.11:
                parents=p
        else:                       #accepting if new one is somehow good
            if f2-f1>=0.12:
                parents=p
        p1+=1
        iterations+=1
        pbar.update(1)

print("Total iterations took: ",max_iterations)
print("prob: ",probability)
print("temp: ",temperature)
best=parents
print("BEST: ",best)
print("fitness of best: ",fitness_function(best))
print("Total distance travelled to visit 312 cities: ",total_distance(best), " km.")
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

plt.scatter(x,y,c="blue",s=3,alpha=.5)
plt.show()

w=0
for w in range(len(best)-1):
    plt.plot(x[w:w+2], y[w:w+2], '-bo')

plt.show()


