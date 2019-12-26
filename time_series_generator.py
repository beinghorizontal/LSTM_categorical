# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 11:37:55 2016

@author: @beinghorizontal

"""

import numpy as np
import pandas as pd



def shift():
    
    for i in range(10):
        shift = np.random.randint(-18,19)
        target.append(shift)

def fill():
    for i in range(10):
        fill = np.random.randint(-14,15)
        inputs.append(fill)

def seq1():
    

    for i in range(10):
        a = np.random.randint(1,4)
        inputs.append(a)
        b = np.random.randint(5,8)
        target.append(b)

def seq2():

    for i in range(10):
        a = np.random.randint(10,14)
        inputs.append(a)
        b = np.random.randint(15,18)
        target.append(b)

def seq3():

    for j in range(10):
        a = np.random.randint(-4,-1)
        inputs.append(a)
        b = np.random.randint(-8,-5)
        target.append(b)

def seq4():

    for j in range(10):
        a = np.random.randint(-14,-10)
        inputs.append(a)
        b = np.random.randint(-18,-15)
        target.append(b)


def populate(sample_size):
    def create():
        
        if num ==1:
            seq1()
        if num ==2:
            seq2()
        if num ==3:
            seq3()
        if num ==4:
            seq4()

    
    for i in range(sample_size):
        num=np.random.randint(1,5)
        create()
        #print('sequence ', num)


def main(size):
    
    
    "step 1 fill targets with 10 random numbers to shift paired sequence by 10 "
    shift()
    
    "step 2. Main bulk of csv file filled with paired sequences"
    populate(sample_size=size)    
    
    "step 3 fill 10 random numbers to match number of rows for targets since targets are 10 steps ahead to begin with"
    fill()

    df1 = pd.DataFrame({'inputs':inputs,'target':target})
    df1.to_csv('wave_memory.csv',header=True,index=False)
    print('last 20 rows ')
    print(df1.tail(20))

inputs = [] #X
target = [] #Y

"For 2000 rows sample_size = 200. Use sample size > 2k so LSTM will have no difficulty to train "
main(size=200)
