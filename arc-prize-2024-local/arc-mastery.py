import time

import os, gc
import sys, pdb
import copy, time
import json, random

import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from pathlib import Path

import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
from colorama import Style, Fore
%matplotlib inline

!ls ../input/*

train1_path = '../input/arc-prize-2024/arc-agi_training_challenges.json'
train2_path = '../input/arc-prize-2024/arc-agi_training_solutions.json'

eval1_path = '../input/arc-prize-2024/arc-agi_evaluation_challenges.json'
eval2_path = '../input/arc-prize-2024/arc-agi_evaluation_solutions.json'

test_path = '../input/arc-prize-2024/arc-agi_test_challenges.json'
sample_path = '../input/arc-prize-2024/sample_submission.json'

# ......................................................................................................
cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

norm = colors.Normalize(vmin=0, vmax=9)
color_list = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "sky", "brown"]

# ......................................................................................................
plt.figure(figsize=(5, 2), dpi=200)
plt.imshow([list(range(10))], cmap=cmap, norm=norm)
plt.xticks(list(range(10)))
plt.yticks([])
plt.show()

# ......................................................................................................

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def plot_pic(x):
    plt.imshow(np.array(x), cmap=cmap, norm=norm)
    plt.show()
    
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def plot_data(task):
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(2, n, figsize=(4*n,8), dpi=200)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_num = 0
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Train-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Train-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1
    for i, t in enumerate(task["test"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Test-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Test-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1
    
    plt.tight_layout()
    plt.show()
    
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def plot_task(task):
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(2, n, figsize=(4*n,8), dpi=200)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_num = 0
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Train-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Train-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1
    for i, t in enumerate(task["test"]):
        t_in = np.array(t["input"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Test-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        fig_num += 1
    
    plt.tight_layout()
    plt.show()  
    
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: hellow_arc
def hellow_arc(n, name_file, values=False):
    '''
    name_file: ('tr' or 'train' or 'training') then (n=0 up to n=399)
    name_file: ('ev' or 'eval' or 'evaluation') then (n=0 up to n=399)
    name_file: ('te' or 'test' or 'testing') then (n=0 up to n=99)
    
    values=False: (Don't print task values)
    values=True: (Print task values)
    '''
    # ...............................................................................
    if (name_file =='tr') or (name_file =='train') or (name_file =='training'):
        
        with open(train1_path,'r') as f:
            train1_name = list(json.load(f).keys())

        with open(train2_path,'r') as f:
            train2_name = list(json.load(f).keys())
        
        with open(train1_path,'r') as f:
            train1_task = list(json.load(f).values())

        with open(train2_path,'r') as f:
            train2_task = list(json.load(f).values())
        
        display(pd.DataFrame(data={'Task Name': train1_name[n], 'Files': 'training_challenges'}, index=[n]))
        
        if (values):
            display(train1_task[n])
        plot_task(train1_task[n])
    
        for i in range(len(train2_task[n])):
            display(pd.DataFrame(data={'Answers for task': train2_name[n], 'Items': i, 'Files': 'training_solutions'}, index=[n]))
            
            if (values):
                display(train2_task[n][i])
            plot_pic(train2_task[n][i])         
    
    # ...............................................................................
    if (name_file =='ev') or (name_file =='eval') or (name_file =='evaluation'):
        
        with open(eval1_path,'r') as f:
            eval1_name = list(json.load(f).keys())

        with open(eval2_path,'r') as f:
            eval2_name = list(json.load(f).keys())
        
        with open(eval1_path,'r') as f:
            eval1_task = list(json.load(f).values())

        with open(eval2_path,'r') as f:
            eval2_task = list(json.load(f).values())
        
        display(pd.DataFrame(data={'Task Name': eval1_name[n], 'Files': 'evaluation_challenges'}, index=[n]))
        
        if (values):
            display(eval1_task[n])
        plot_task(eval1_task[n])
    
        for i in range(len(eval2_task[n])):
            display(pd.DataFrame(data={'Answers for task': eval2_name[n], 'Items': i, 'Files': 'evaluation_solutions'}, index=[n]))
            
            if (values):
                display(eval2_task[n][i])
            plot_pic(eval2_task[n][i])    

    # ...............................................................................
    if (name_file =='te') or (name_file =='test') or (name_file =='testing'):
        
        with open(test_path,'r') as f:
            test_name = list(json.load(f).keys())   
    
        with open(test_path,'r') as f:
            test_task = list(json.load(f).values())
           
        display(pd.DataFrame(data={'Task Name': test_name[n], 'Files': 'test_challenges'}, index=[n]))
        
        if (values):
            display(test_task[n])
        plot_task(test_task[n])
        
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: get_arc
def get_arc(n, name_file):
    '''
    name_file: ('tr' or 'train' or 'training') then (n=0 up to n=399)
    name_file: ('ev' or 'eval' or 'evaluation') then (n=0 up to n=399)
    name_file: ('te' or 'test' or 'testing') then (n=0 up to n=99)
    '''
    # ...............................................................................
    if (name_file =='tr') or (name_file =='train') or (name_file =='training'):
        
        with open(train1_path,'r') as f:
            train1_task = list(json.load(f).values())

        with open(train2_path,'r') as f:
            train2_task = list(json.load(f).values())
            
        task = train1_task[n] 
        answer = train2_task[n] 
        
        inp = []
        for i in range(len(task['test'])):
            inp_test = np.array(task['test'][i]['input'])
            inp.append(inp_test)
    
    # ...............................................................................
    if (name_file =='ev') or (name_file =='eval') or (name_file =='evaluation'):
        
        with open(eval1_path,'r') as f:
            eval1_task = list(json.load(f).values())

        with open(eval2_path,'r') as f:
            eval2_task = list(json.load(f).values())
        
        task = eval1_task[n] 
        answer = eval2_task[n] 
        
        inp = []
        for i in range(len(task['test'])):
            inp_test = np.array(task['test'][i]['input'])
            inp.append(inp_test)

    # ...............................................................................
    if (name_file =='te') or (name_file =='test') or (name_file =='testing'):
        
        with open(test_path,'r') as f:
            test_task = list(json.load(f).values())
            
        task = test_task[n] 
        answer = [] 
        
        inp = []
        for i in range(len(task['test'])):
            inp_test = np.array(task['test'][i]['input'])
            inp.append(inp_test)
        
    # ...............................................................................
   
    return inp[0], task, answer[0]
        
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 

hellow_arc(185,'train', True)
hellow_arc(315,'eval')
hellow_arc(82,'test')

version = []
solvers = []

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 0 Rotation & Flip
def f0000(inp, task):
    '''
    This function changes the shape of a task via Rotation & Flip.
    '''  
    clist = []
    clist.append(inp)                                    # v0 Same
    clist.append(np.rot90(inp, 1))                       # v1 rot90
    clist.append(np.rot90(inp, 2))                       # v2 rot180
    clist.append(np.rot90(inp, 3))                       # v3 rot270
    clist.append(np.flip(inp, axis=0))                   # v4 flip_0
    clist.append(np.flip(inp, axis=1))                   # v5 flip_1
    clist.append(np.flip(np.rot90(inp, 1), axis=0))      # v6 flip_0(rot90)
    clist.append(np.flip(np.rot90(inp, 3), axis=0))      # v7 flip_0(rot270)
    
    return clist

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
version.append(8)
solvers.append(f0000)

inp = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

plot_pic(inp) 

clist = f0000(inp, [])

for x in clist:
    plot_pic(x)

hellow_arc(238,'train')

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 
def ft1(inp, task):
    '''
    This function sorts and draws all repetitions of each color.
    '''  
    uinp = np.unique(inp, return_counts= True)[0]
    n_uinp = np.unique(inp, return_counts= True)[1]
        
    uinp_sort = uinp[np.argsort(n_uinp)]
    n_uinp_sort = np.sort(n_uinp)
    
    # .......................................................
    created = np.zeros((len(uinp_sort), n_uinp_sort[-1]))
    
    for c in range(len(uinp_sort)): 
        for n in range(n_uinp_sort[c]):
            created[c, n] = uinp_sort[c]
            
    created = np.rot90(created, 3)
    return created

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

inp, task, answer = get_arc(238,'train')
created = ft1(inp, task)

print(np.array_equal(created, answer))
plot_pic(created) 

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 1 SortUniqueColor
def f0001(inp, task):
    '''
    This function sorts and draws all repetitions of each color.
    '''  
    clist = []

    uinp = np.unique(inp, return_counts= True)[0]
    n_uinp = np.unique(inp, return_counts= True)[1]
        
    uinp_sort = uinp[np.argsort(n_uinp)]
    n_uinp_sort = np.sort(n_uinp)
    
    # ............................................................. 
    inpz = np.array(task['train'][0]['input'])
    outz = np.array(task['train'][0]['output'])
            
    uinpz = np.unique(inpz, return_counts= True)[0]
    uoutz = np.unique(outz, return_counts= True)[0]
            
    not_inpz = [f for f in uoutz if f not in uinpz]
    not_outz = [f for f in uinpz if f not in uoutz]

    # ............................................................. 
    try:
        created = np.full((len(uinp_sort), n_uinp_sort[-1]), not_inpz[0])
    
        for c in range(len(uinp_sort)): 
            for n in range(n_uinp_sort[c]):
                created[c, n] = uinp_sort[c]
            
        clist = f0000(created, [])   
        
    except:
        for n in range(8):
            clist.append([])      
 
    # ............................................................. 
    return clist

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
version.append(8)
solvers.append(f0001)

inp, task, answer = get_arc(238,'train')
clist = f0001(inp, task)

for x in clist:
    if (len(x)!= 0):  # For when is not equal to []
        print(np.array_equal(x, answer))
        plot_pic(x) 

hellow_arc(375,'eval')

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def ft2(inp, task):
    '''
    This function rotates the images and then concatenates them together.
    '''  
    pr1 = inp
    pr2 = np.rot90(inp, 1) 
    pr3 = np.concatenate((pr1, pr2), axis=1)
    
    # ................................................
    prn1 = np.rot90(inp, 2) 
    prn2 = np.rot90(inp, 3)
    prn3 = np.concatenate((prn1, prn2), axis=1)
    # ................................................
    
    created = np.concatenate((pr3, prn3), axis=0)
    return created

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


inp, task, answer = get_arc(375,'eval')
created = ft2(inp, task)

print(np.array_equal(created, answer))
plot_pic(created) 


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 2   Catenat_2ele_2ele
def f0002(inp, task):
    '''
    This function concatenates two versions (2+2) of the same image with different rotations.
    '''   
    clist = []
    alist = f0000(inp, [])
    
    # .............................................................  
    for x in alist:
        for y in alist:
            
            try:
                pred = np.concatenate((x, y), axis=0)
                clist.append(pred) 
            except:
                clist.append([])    
 

            try:
                pred = np.concatenate((x, y), axis=1)
                clist.append(pred) 
            except:
                clist.append([])     
                
       
            try:
                pred = np.concatenate((x, y), axis=0)
                pred = np.concatenate((pred, pred), axis=1)
                clist.append(pred) 
            except:
                clist.append([])      
  

            try:
                pred = np.concatenate((x, y), axis=1)
                pred = np.concatenate((pred, pred), axis=0)
                clist.append(pred) 
            except:
                clist.append([])   
                
 
            try:
                pred1 = np.concatenate((x, y), axis=0)
                pred2 = np.concatenate((y, x), axis=0)
                pred  = np.concatenate((pred1, pred2), axis=1)
                clist.append(pred) 
            except:
                clist.append([])      
  

            try:
                pred1 = np.concatenate((x, y), axis=1)
                pred2 = np.concatenate((y, x), axis=1)
                pred  = np.concatenate((pred1, pred2), axis=0)
                clist.append(pred) 
            except:
                clist.append([])   # v0 ~ v383

            # .........................................................
            
            try:
                pred1 = np.concatenate((x, y), axis=0)
                pred2 = np.flip(pred1, axis=0)
                pred  = np.concatenate((pred1, pred2), axis=1)
                clist.append(pred) 
            except:
                clist.append([])      
                
            
            try:
                pred1 = np.concatenate((x, y), axis=0)
                pred2 = np.flip(pred1, axis=1)
                pred  = np.concatenate((pred1, pred2), axis=1)
                clist.append(pred) 
            except:
                clist.append([])    

                
            try:
                pred1 = np.concatenate((x, y), axis=1)
                pred2 = np.flip(pred1, axis=0)
                pred  = np.concatenate((pred1, pred2), axis=0)
                clist.append(pred) 
            except:
                clist.append([])   
                
            
            try:
                pred1 = np.concatenate((x, y), axis=1)
                pred2 = np.flip(pred1, axis=1)
                pred  = np.concatenate((pred1, pred2), axis=0)
                clist.append(pred) 
            except:
                clist.append([])   
            
            
            try:
                pred1 = np.concatenate((x, y), axis=0)
                pred2 = np.flip(pred1, axis=0)
                pred  = np.concatenate((pred2, pred1), axis=1)
                clist.append(pred) 
            except:
                clist.append([])      
                
                
            try:
                pred1 = np.concatenate((x, y), axis=0)
                pred2 = np.flip(pred1, axis=1)
                pred  = np.concatenate((pred2, pred1), axis=1)
                clist.append(pred) 
            except:
                clist.append([]) 

                
            try:
                pred1 = np.concatenate((x, y), axis=1)
                pred2 = np.flip(pred1, axis=0)
                pred  = np.concatenate((pred2, pred1), axis=0)
                clist.append(pred) 
            except:
                clist.append([])   
                
                
            try:
                pred1 = np.concatenate((x, y), axis=1)
                pred2 = np.flip(pred1, axis=1)
                pred  = np.concatenate((pred2, pred1), axis=0)
                clist.append(pred) 
            except:
                clist.append([])   # v384 ~ v895
                
    # ............................................................. 
    return clist

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 
version.append(896)
solvers.append(f0002)

inp, task, answer = get_arc(375,'eval')
clist = f0002(inp, task)
print('len clist:', len(clist))

for x in clist:
    if (np.array_equal(x, answer)):
        print('\nThis is correct:')
        plot_pic(x) 
        break

for i in range(5):
    n = random.randrange(len(clist))
    
    if (len(clist[n])!= 0):  # For when is not equal to []
        plot_pic(clist[n])    

hellow_arc(66,'train')

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 3  SelectZone
def f0003(inp, task):
    '''
    This function returns the selected zone.
    '''   
    clist = []
    ii = inp.shape[0]
    jj = inp.shape[1]
    
    uinp = np.unique(inp, return_counts= True)[0]
    n_uinp = np.unique(inp, return_counts= True)[1]
    
    uinp_min = uinp[np.argsort(n_uinp)[0]]
    uinp_max = uinp[np.argsort(n_uinp)[-1]]
    
    # ............................................................. 
    try:
        clist.append(np.array([[inp[0,0]]]))
    except:
        clist.append([])  # v0

    # ............................................................. 
    try:
        clist.append(np.array([[inp[-1,-1]]]))
    except:
        clist.append([])  # v1

    # ............................................................. 
    try:
        clist.append(np.array([[inp[0,-1]]]))
    except:
        clist.append([])  # v2

    # ............................................................. 
    try:
        clist.append(np.array([[inp[-1,0]]]))
    except:
        clist.append([])  # v3

    # ............................................................. 
    try:
        clist.append(np.array([inp[:,0]]))
    except:
        clist.append([])  # v4

    # ............................................................. 
    try:
        clist.append(np.array([inp[0,:]]))
    except:
        clist.append([])  # v5

    # ............................................................. 
    try:
        clist.append(np.array([inp[:,-1]]))
    except:
        clist.append([])  # v6

    # ............................................................. 
    try:
        clist.append(np.array([inp[-1,:]]))
    except:
        clist.append([])  # v7

    # ............................................................. 
    try:
        clist.append(np.array(inp[:2,:2]))
    except:
        clist.append([])  # v8

    # ............................................................. 
    try:
        clist.append(np.array(inp[:2,:]))
    except:
        clist.append([])  # v9

    # .............................................................  
    try:
        clist.append(np.array(inp[:,:2]))
    except:
        clist.append([])  # v10

    # ............................................................. 
    try:
        clist.append(np.array(inp[:ii//2 ,:]))
    except:
        clist.append([])  # v11

    # ............................................................. 
    try:
        clist.append(np.array(inp[ii//2: ,:]))
    except:
        clist.append([])  # v12

    # ............................................................. 
    try:
        clist.append(np.array(inp[: , :jj//2]))
    except:
        clist.append([])  # v13

    # ............................................................. 
    try:
        clist.append(np.array(inp[: , jj//2:]))
    except:
        clist.append([])  # v14

    # ............................................................. 
    try:
        clist.append(np.array(inp[:ii//2 , :jj//2]))
    except:
        clist.append([])  # v15

    # ............................................................. 
    try:
        clist.append(np.array(inp[ii//2: , :jj//2]))
    except:
        clist.append([])  # v16

    # .............................................................
    try:
        clist.append(np.array(inp[:ii//2 , jj//2:]))
    except:
        clist.append([])  # v17

    # ............................................................. 
    try:
        clist.append(np.array(inp[ii//2: , jj//2:]))
    except:
        clist.append([])  # v18

    # ............................................................. 
    try:
        clist.append(np.array(inp[-2:,-2:]))
    except:
        clist.append([])  # v19

    # ............................................................. 
    try:
        clist.append(np.array(inp[-2:,:]))
    except:
        clist.append([])  # v20

    # ............................................................. 
    try:
        clist.append(np.array(inp[:,-2:]))
    except:
        clist.append([])  # v21

    # ............................................................. 
    try:
        clist.append(np.array(inp[:3,:3]))
    except:
        clist.append([])  # v22

    # ............................................................. 
    try:
        clist.append(np.array(inp[:3,:]))
    except:
        clist.append([])  # v23

    # ............................................................. 
    try:
        clist.append(np.array(inp[:,:3]))
    except:
        clist.append([])  # v24

    # .............................................................         
    try:
        clist.append(np.array(inp[:ii//3 , :]))
    except:
        clist.append([])  # v25

    # .............................................................         
    try:
        clist.append(np.array(inp[: , :jj//3]))
    except:
        clist.append([])  # v26

    # ............................................................. 
    try:
        clist.append(np.array(inp[:ii//3 , :jj//3]))
    except:
        clist.append([])  # v27

    # ............................................................. 
    try:
        clist.append(np.array(inp[-3:,-3:]))
    except:
        clist.append([])  # v28

    # ............................................................. 
    try:
        clist.append(np.array(inp[-3:,:]))
    except:
        clist.append([])  # v29

    # ............................................................. 
    try:
        clist.append(np.array(inp[:,-3:]))
    except:
        clist.append([])  # v30

    # ............................................................. 
    try:
        leni = inp.shape[0]//2
        created = np.full((inp.shape), uinp_max)
        created[leni, :] = inp[leni, :] 
        
        clist.append(created)
    except:
        clist.append([])  # v31

    # ............................................................. 
    try:
        leni = inp.shape[0]//2
        created = np.full((inp.shape), 0)
        created[leni, :] = inp[leni, :] 
        
        clist.append(created)
    except:
        clist.append([])  # v32

    # ............................................................. 
    try:
        lenj = inp.shape[1]//2
        created = np.full((inp.shape), uinp_max)
        created[:, lenj] = inp[:, lenj] 
    
        clist.append(created)
    except:
        clist.append([])  # v33
        
    # ............................................................. 
    try:
        lenj = inp.shape[1]//2
        created = np.full((inp.shape), 0)
        created[:, lenj] = inp[:, lenj] 
    
        clist.append(created)
    except:
        clist.append([])  # v34       
        
    # .............................................................         
    try:
        leni = inp.shape[0]//2
        lenj = inp.shape[1]//2
        created = np.full((inp.shape), uinp_max)
        created[leni, :] = inp[leni, :] 
        created[:, lenj] = inp[:, lenj]
        
        clist.append(created)
    except:
        clist.append([])  # v35

    # .............................................................         
    try:
        leni = inp.shape[0]//2
        lenj = inp.shape[1]//2
        created = np.full((inp.shape), 0)
        created[leni, :] = inp[leni, :] 
        created[:, lenj] = inp[:, lenj]
        
        clist.append(created)
    except:
        clist.append([])  # v36
        
    # .............................................................         
    try:
        leni = inp.shape[0]//2
        lenj = inp.shape[1]//2
        created = np.full((inp.shape), uinp_max)
        created[leni, lenj] = inp[leni, lenj] 
        
        clist.append(created)
    except:
        clist.append([])  # v37

    # .............................................................         
    try:
        leni = inp.shape[0]//2
        lenj = inp.shape[1]//2
        created = np.full((inp.shape), 0)
        created[leni, lenj] = inp[leni, lenj] 
        
        clist.append(created)
    except:
        clist.append([])  # v38        
           
    # .............................................................         
    try:
        created = np.full((inp.shape), uinp_max)
        
        for x in range(inp.shape[0]):
            for y in range(inp.shape[1]):
                if (x == y):
                    created[x, y] = inp[x, y] 
                    
        clist.append(created)
    except:
        clist.append([])  # v39
        
    # .............................................................         
    try:
        created = np.full((inp.shape), 0)
        
        for x in range(inp.shape[0]):
            for y in range(inp.shape[1]):
                if (x == y):
                    created[x, y] = inp[x, y] 
                    
        clist.append(created)
    except:
        clist.append([])  # v40        
        
    # .............................................................         
    try:
        created = np.full((inp.shape), uinp_max)
        
        for x in range(inp.shape[0]):
            for y in range(inp.shape[1]):
                if (x == y):
                    created[x, -y-1] = inp[x, -y-1] 
                    
        clist.append(created)
    except:
        clist.append([])  # v41
        
    # .............................................................         
    try:
        created = np.full((inp.shape), 0)
        
        for x in range(inp.shape[0]):
            for y in range(inp.shape[1]):
                if (x == y):
                    created[x, -y-1] = inp[x, -y-1] 
                    
        clist.append(created)
    except:
        clist.append([])  # v42        
        
    # ............................................................. 
    try:
        created = np.full((inp.shape), uinp_max)
        
        for x in range(inp.shape[0]):
            for y in range(inp.shape[1]):
                if (x == y):
                    created[x, y] = inp[x, y]
                    created[x, -y-1] = inp[x, -y-1] 
                    
        clist.append(created)
    except:
        clist.append([])  # v43
        
    # .............................................................         
    try:
        created = np.full((inp.shape), 0)
        
        for x in range(inp.shape[0]):
            for y in range(inp.shape[1]):
                if (x == y):
                    created[x, y] = inp[x, y]
                    created[x, -y-1] = inp[x, -y-1] 
                    
        clist.append(created)
    except:
        clist.append([])  # v44      
        
    # .............................................................     
    try:
        leni = inp.shape[0]//2
        clist.append(np.array([inp[leni, :]]))
    except:
        clist.append([])  # v45

    # ............................................................. 
    try:
        lenj = inp.shape[1]//2
        clist.append(np.array([inp[:, lenj]]))
    except:
        clist.append([])  # v46

    # .............................................................         
    try:
        leni = inp.shape[0]//2
        lenj = inp.shape[1]//2
        clist.append(np.array([[inp[leni, lenj]]]))
    except:
        clist.append([])  # v47

    # .............................................................         
    try:
        created = np.full((inp.shape[0], 1), uinp_max)
        
        for x in range(inp.shape[0]):
            for y in range(inp.shape[1]):
                if (x == y):
                    created[x, 0] = inp[x, y] 
                    
        clist.append(created)
    except:
        clist.append([])  # v48
        
    # .............................................................         
    try:
        created = np.full((inp.shape[0], 1), 0)
        
        for x in range(inp.shape[0]):
            for y in range(inp.shape[1]):
                if (x == y):
                    created[x, 0] = inp[x, y] 
                    
        clist.append(created)
    except:
        clist.append([])  # v49       
        
    # .............................................................         
    try:
        created = np.full((inp.shape[0], 1), uinp_max)
        
        for x in range(inp.shape[0]):
            for y in range(inp.shape[1]):
                if (x == y):
                    created[x, 0] = inp[x, -y-1] 
                    
        clist.append(created)
    except:
        clist.append([])  # v50
        
    # .............................................................         
    try:
        created = np.full((inp.shape[0], 1), 0)
        
        for x in range(inp.shape[0]):
            for y in range(inp.shape[1]):
                if (x == y):
                    created[x, 0] = inp[x, -y-1] 
                    
        clist.append(created)
    except:
        clist.append([])  # v51
       
    # .............................................................    
    return clist

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 
version.append(52)
solvers.append(f0003)        
               
inp, task, answer = get_arc(66,'train')
clist = f0003(inp, task)
print('len clist:', len(clist))

for x in clist:
    if (np.array_equal(x, answer)):
        print('\nThis is correct:')
        plot_pic(x) 
        break

for i in range(5):
    n = random.randrange(len(clist))
    
    if (len(clist[n])!= 0):  # For when is not equal to []
        plot_pic(clist[n])    

hellow_arc(115,'eval')

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 4  SudokuPuzzles
def f0004(inp, task):
    '''
    This function works like Sudoku puzzles.
    '''  
    clist = []
    
    # .............................................................
    inpz = np.array(task['train'][0]['input'])
    outz = np.array(task['train'][0]['output'])
            
    uinpz = np.unique(inpz, return_counts= True)[0]
    uoutz = np.unique(outz, return_counts= True)[0]
            
    not_inpz = [f for f in uoutz if f not in uinpz]
    not_outz = [f for f in uinpz if f not in uoutz]

    # ............................................................. 
    try:
        created = inp.copy()
    
        uinp = np.unique(inp, return_counts= True)[0]
        uinp = [f for f in uinp if (f!= not_outz[0])]
    
        for n in range(10):
            for x in range(inp.shape[0]):
                for y in range(inp.shape[1]):
            
                    if (created[x, y] == not_outz[0]):
                
                        setx = created[x, :]
                        usetx = np.unique(setx)
                        if (not_outz[0] in usetx):
                            if (len(usetx) == len(uinp)):
                                select = [f for f in uinp if (f not in usetx)]
                                created[x, :][created[x, :] ==  not_outz[0]] = select[0]
                        
                        sety = created[:, y]
                        usety = np.unique(sety)
                        if (not_outz[0] in usety):
                            if (len(usety) == len(uinp)):
                                select = [f for f in uinp if (f not in usety)]
                                created[:, y][created[:, y] ==  not_outz[0]] = select[0]                

        clist.append(created)   
    except:
        clist.append([])    # v17          
         
    # .............................................................  
    return clist

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 
version.append(1)
solvers.append(f0004)

inp, task, answer = get_arc(115,'eval')
clist = f0004(inp, task)

for x in clist:
    if (len(x)!= 0):  # For when is not equal to []
        print(np.array_equal(x, answer))
        plot_pic(x) 
hellow_arc(286,'train')

hellow_arc(16,'train')


hellow_arc(304,'train')


hellow_arc(383,'eval')


# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def ft3(inp, task):
    '''
    This function performs "Carpet Repair" by flipping and rotating.
    ''' 
    inpz = np.array(task['train'][0]['input'])
    outz = np.array(task['train'][0]['output'])
            
    uinpz = np.unique(inpz, return_counts= True)[0]
    uoutz = np.unique(outz, return_counts= True)[0]
            
    not_inpz = [f for f in uoutz if f not in uinpz]
    not_outz = [f for f in uinpz if f not in uoutz]

    # ............................................................. 
    try: 
        created = inp.copy()
        c_change = not_outz[0]
        seet = np.flip(np.rot90(inp, 1), axis=0)
    
        for x in range(inp.shape[0]):
            for y in range(inp.shape[1]):
            
                if (created[x, y]== c_change) and (seet[x, y]!= c_change):
                    created[x, y] = seet[x, y]
    except:
        created = []
        
    return created

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def ft4(inp, task):
    '''
    This function performs "Carpet Repair" by flipping and rotating.
    ''' 
    inpz = np.array(task['train'][0]['input'])
    outz = np.array(task['train'][0]['output'])
            
    uinpz = np.unique(inpz, return_counts= True)[0]
    uoutz = np.unique(outz, return_counts= True)[0]
            
    not_inpz = [f for f in uoutz if f not in uinpz]
    not_outz = [f for f in uinpz if f not in uoutz]

    # ............................................................. 
    try: 
        created = inp.copy()
        c_change = not_outz[0]
        seet = np.flip(np.rot90(inp, 3), axis=0)
    
        for x in range(inp.shape[0]):
            for y in range(inp.shape[1]):
            
                if (created[x, y]== c_change) and (seet[x, y]!= c_change):
                    created[x, y] = seet[x, y]           
    except:
        created = []
        
    return created

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


# Example 9
inp, task, answer = get_arc(286,'train')
created = ft3(inp, task)

print(np.array_equal(created, answer))
plot_pic(created) 


# Example 9
inp, task, answer = get_arc(286,'train')
created = ft4(inp, task)

print(np.array_equal(created, answer))
plot_pic(created) 


# Example 10-1
inp, task, answer = get_arc(16,'train')
created = ft3(inp, task)

print(np.array_equal(created, answer))
plot_pic(created)     

# Example 10-2
inp, task, answer = get_arc(16,'train')
created = ft4(inp, task)

print(np.array_equal(created, answer))
plot_pic(created) 

# Example 11-1
inp, task, answer = get_arc(304,'train')
created = ft3(inp, task)

print(np.array_equal(created, answer))
plot_pic(created) 


# Example 11-2
inp, task, answer = get_arc(304,'train')
created = ft4(inp, task)

print(np.array_equal(created, answer))
plot_pic(created) 

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 5  CarpetRepair_1
def f0005(inp, task):
    '''
    This function performs a "Carpet Repair" and after completing the initial shape, returns both the whole shape and the completed part.
    '''  
    clist = []
    
    # .............................................................
    inpz = np.array(task['train'][0]['input'])
    outz = np.array(task['train'][0]['output'])
            
    uinpz = np.unique(inpz, return_counts= True)[0]
    uoutz = np.unique(outz, return_counts= True)[0]
            
    not_inpz = [f for f in uoutz if f not in uinpz]
    not_outz = [f for f in uinpz if f not in uoutz]

    # ............................................................. 
    try: 
        created = inp.copy()
        c_change = not_outz[0]
        
        uinp = np.unique(inp)
        wcolor = np.argwhere(inp == c_change)
      
        wrow, wcol = [], []
        for w in wcolor:
            wrow.append(w[0])
            wcol.append(w[1])
        
        minr = min(wrow)
        maxr = max(wrow)
        minc = min(wcol)
        maxc = max(wcol) 
    
        # .........................................................
        seet = np.flip(np.rot90(inp, 1), axis=0)
    
        for x in range(inp.shape[0]):
            for y in range(inp.shape[1]):
            
                if (created[x, y]== c_change) and (seet[x, y]!= c_change):
                    created[x, y] = seet[x, y]     
                        
        # .........................................................
        for i in range(created.shape[0]):
            if c_change in created[i, :]:  
            
                rowi = created[i, :]
                arg_change   = np.argwhere(rowi == c_change).ravel()
                arg_unchange = np.argwhere(rowi != c_change).ravel()
              
                changei   = rowi[arg_change]
                unchangei = rowi[arg_unchange]
            
                for ii in range(created.shape[0]):
                
                    rowii = created[ii, :]
                    changeii   = rowii[arg_change]
                    unchangeii = rowii[arg_unchange]
                
                    if (np.array_equal(unchangei, unchangeii)):
                        created[i, :] = rowii
                
        for j in range(created.shape[1]):
            if c_change in created[:, j]:  
            
                colj = created[:, j]
                arg_change   = np.argwhere(colj == c_change).ravel()
                arg_unchange = np.argwhere(colj != c_change).ravel()
              
                changej   = colj[arg_change]
                unchangej = colj[arg_unchange]
            
                for jj in range(created.shape[1]):
                
                    coljj = created[:, jj]
                    changejj   = coljj[arg_change]
                    unchangejj = coljj[arg_unchange]
                
                    if (np.array_equal(unchangej, unchangejj)):
                        created[:, j] = coljj
                    
        # .........................................................
        for i in range(created.shape[0]):
            if c_change in created[i, :]:  
            
                rowi = created[i, :]
                arg_change   = np.argwhere(rowi == c_change).ravel()
                arg_unchange = np.argwhere(rowi != c_change).ravel()
              
                changei   = rowi[arg_change]
                unchangei = rowi[arg_unchange]
            
                for ii in range(created.shape[0]):
                
                    rowii = created[ii, :]
                    changeii   = rowii[arg_change]
                    unchangeii = rowii[arg_unchange]
                
                    if (np.array_equal(unchangei, unchangeii)):
                        created[i, :] = rowii
                
        for j in range(created.shape[1]):
            if c_change in created[:, j]:  
            
                colj = created[:, j]
                arg_change   = np.argwhere(colj == c_change).ravel()
                arg_unchange = np.argwhere(colj != c_change).ravel()
              
                changej   = colj[arg_change]
                unchangej = colj[arg_unchange]
            
                for jj in range(created.shape[1]):
                
                    coljj = created[:, jj]
                    changejj   = coljj[arg_change]
                    unchangejj = coljj[arg_unchange]
                
                    if (np.array_equal(unchangej, unchangejj)):
                        created[:, j] = coljj
                    
        # .............................................................          
        clist.append(created)   
        created = created[minr:maxr+1 , minc:maxc+1]  
        clist.append(created) 
    except:
        clist.append([])
        clist.append([])

    # .................................................................  
    return clist

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 
version.append(2)
solvers.append(f0005)               


# Example 9
inp, task, answer = get_arc(286,'train')
clist = f0005(inp, task)

for x in clist:
    if (np.array_equal(x, answer)):
        print('\nThis is correct:')
        plot_pic(x) 

# Example 10
inp, task, answer = get_arc(16,'train')
clist = f0005(inp, task)

for x in clist:
    if (np.array_equal(x, answer)):
        print('\nThis is correct:')
        plot_pic(x) 


# Example 11
inp, task, answer = get_arc(304,'train')
clist = f0005(inp, task)

for x in clist:
    if (np.array_equal(x, answer)):
        print('\nThis is correct:')
        plot_pic(x) 

# Example 12
inp, task, answer = get_arc(383,'eval')
clist = f0005(inp, task)

for x in clist:
    if (np.array_equal(x, answer)):
        print('\nThis is correct:')
        plot_pic(x) 

hellow_arc(391,'eval')

# Example 13
inp, task, answer = get_arc(391,'eval')
clist = f0005(inp, task)

for x in clist:
    if (np.array_equal(x, answer)):
        print('\nThis is correct:')
        plot_pic(x)  
    else:
        print('\nThis is not correct:')
        plot_pic(x)   


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 6  CarpetRepair_2
def f0006(inp, task):
    '''
    This function performs a "Carpet Repair" and after completing the initial shape, returns both the whole shape and the completed part.
    '''  
    clist = []
    
    # .............................................................
    inpz = np.array(task['train'][0]['input'])
    outz = np.array(task['train'][0]['output'])
            
    uinpz = np.unique(inpz, return_counts= True)[0]
    uoutz = np.unique(outz, return_counts= True)[0]
            
    not_inpz = [f for f in uoutz if f not in uinpz]
    not_outz = [f for f in uinpz if f not in uoutz]

    # ............................................................. 
    try: 
        created = inp.copy()
        c_change = not_outz[0]
        
        uinp = np.unique(inp)
        wcolor = np.argwhere(inp == c_change)
      
        wrow, wcol = [], []
        for w in wcolor:
            wrow.append(w[0])
            wcol.append(w[1])
        
        minr = min(wrow)
        maxr = max(wrow)
        minc = min(wcol)
        maxc = max(wcol) 
    
        # .........................................................
        for i in range(created.shape[0]):
            if c_change in created[i, :]:  
            
                rowi = created[i, :]
                arg_change   = np.argwhere(rowi == c_change).ravel()
                arg_unchange = np.argwhere(rowi != c_change).ravel()
              
                changei   = rowi[arg_change]
                unchangei = rowi[arg_unchange]
            
                for ii in range(created.shape[0]):
                
                    rowii = created[ii, :]
                    changeii   = rowii[arg_change]
                    unchangeii = rowii[arg_unchange]
                
                    if (np.array_equal(unchangei, unchangeii)):
                        created[i, :] = rowii
                
        for j in range(created.shape[1]):
            if c_change in created[:, j]:  
            
                colj = created[:, j]
                arg_change   = np.argwhere(colj == c_change).ravel()
                arg_unchange = np.argwhere(colj != c_change).ravel()
              
                changej   = colj[arg_change]
                unchangej = colj[arg_unchange]
            
                for jj in range(created.shape[1]):
                
                    coljj = created[:, jj]
                    changejj   = coljj[arg_change]
                    unchangejj = coljj[arg_unchange]
                
                    if (np.array_equal(unchangej, unchangejj)):
                        created[:, j] = coljj
                    
        # .........................................................
        for i in range(created.shape[0]):
            if c_change in created[i, :]:  
            
                rowi = created[i, :]
                arg_change   = np.argwhere(rowi == c_change).ravel()
                arg_unchange = np.argwhere(rowi != c_change).ravel()
              
                changei   = rowi[arg_change]
                unchangei = rowi[arg_unchange]
            
                for ii in range(created.shape[0]):
                
                    rowii = created[ii, :]
                    changeii   = rowii[arg_change]
                    unchangeii = rowii[arg_unchange]
                
                    if (np.array_equal(unchangei, unchangeii)):
                        created[i, :] = rowii
                
        for j in range(created.shape[1]):
            if c_change in created[:, j]:  
            
                colj = created[:, j]
                arg_change   = np.argwhere(colj == c_change).ravel()
                arg_unchange = np.argwhere(colj != c_change).ravel()
              
                changej   = colj[arg_change]
                unchangej = colj[arg_unchange]
            
                for jj in range(created.shape[1]):
                
                    coljj = created[:, jj]
                    changejj   = coljj[arg_change]
                    unchangejj = coljj[arg_unchange]
                
                    if (np.array_equal(unchangej, unchangejj)):
                        created[:, j] = coljj
         
        # .........................................................
        seet = np.flip(np.rot90(inp, 1), axis=0)
    
        for x in range(inp.shape[0]):
            for y in range(inp.shape[1]):
            
                if (created[x, y]== c_change) and (seet[x, y]!= c_change):
                    created[x, y] = seet[x, y]     
                          
    # .................................................................           
        clist.append(created)   
        created = created[minr:maxr+1 , minc:maxc+1]  
        clist.append(created) 
    except:
        clist.append([])
        clist.append([])

    # .................................................................  
    return clist

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 
version.append(2)
solvers.append(f0006)               


# Example 13
inp, task, answer = get_arc(391,'eval')
clist = f0006(inp, task)

for x in clist:
    if (np.array_equal(x, answer)):
        print('\nThis is correct:')
        plot_pic(x)  


hellow_arc(376,'train')

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def ft5(inp, task):
    '''
    This function removes identical columns and identical rows.
    ''' 
    try: 
        ilist = []
        if (inp.shape[0] > 1):
            for i in range(1, inp.shape[0]):
                if all(inp[i-1, :] == inp[i, :]):
                    ilist.append(i-1)
                    
        jlist = []
        if (inp.shape[1] > 1):
            for j in range(1, inp.shape[1]):
                if all(inp[:, j-1] == inp[:, j]):
                    jlist.append(j-1)
                
        created = inp.copy()
        created = np.delete(created, ilist, 0)
        created = np.delete(created, jlist, 1)           
        
    except:
        created = []
        
    return created

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Example 14
inp, task, answer = get_arc(376,'train')
created = ft5(inp, task)

print(np.array_equal(created, answer))
plot_pic(created) 

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 7 Removes Identical Cols&Rows
def f0007(inp, task):
    '''
    This function removes identical columns and identical rows.
    '''   
    alist = []
    clist = []
    
    uinp = np.unique(inp, return_counts= True)[0]
    n_uinp = np.unique(inp, return_counts= True)[1]
    
    uinp_min = uinp[np.argsort(n_uinp)[0]] 
    uinp_max = uinp[np.argsort(n_uinp)[-1]]
     
    # .............................................................   
    try:
        ilist = []
        if (inp.shape[0] > 1):
            for i in range(1, inp.shape[0]):
                if all(inp[i-1, :] == inp[i, :]):
                    ilist.append(i-1)
                    
        jlist = []
        if (inp.shape[1] > 1):
            for j in range(1, inp.shape[1]):
                if all(inp[:, j-1] == inp[:, j]):
                    jlist.append(j-1)
                
        created = inp.copy()
        created = np.delete(created, ilist, 0)
        created = np.delete(created, jlist, 1)              

        alist.append(created) 
    except:
        alist.append([])    # delete identical rows&cols
           
    # .............................................................   
    try:
        ilist = []
        if (inp.shape[0] > 1):
            for i in range(1, inp.shape[0]):
                if all(inp[i-1, :] == inp[i, :]):
                    ilist.append(i-1)
                
        created = inp.copy()
        created = np.delete(created, ilist, 0)
        
        alist.append(created)
        
    except:
        alist.append([])    # delete identical rows
           
    # .............................................................       
    try:                
        jlist = []
        if (inp.shape[1] > 1):
            for j in range(1, inp.shape[1]):
                if all(inp[:, j-1] == inp[:, j]):
                    jlist.append(j-1)
                
        created = inp.copy()
        created = np.delete(created, jlist, 1)    

        alist.append(created)
        
    except:
        alist.append([])    # delete identical cols
           
    # .............................................................   
    try:
        ilist = []
        if (inp.shape[0] > 1):
            for i in range(1, inp.shape[0]):
                if all(inp[i-1, :] == uinp[0]):
                    ilist.append(i-1)
                    
        jlist = []
        if (inp.shape[1] > 1):
            for j in range(1, inp.shape[1]):
                if all(inp[:, j-1] == uinp[0]):
                    jlist.append(j-1)
                
        created = inp.copy()
        created = np.delete(created, ilist, 0)
        created = np.delete(created, jlist, 1)     

        alist.append(created)
        
    except:
        alist.append([])    # delete all-Zero rows&cols
               
    # .............................................................   
    try:
        ilist = []
        if (inp.shape[0] > 1):
            for i in range(1, inp.shape[0]):
                if all(inp[i-1, :] == uinp_max):
                    ilist.append(i-1)
                    
        jlist = []
        if (inp.shape[1] > 1):
            for j in range(1, inp.shape[1]):
                if all(inp[:, j-1] == uinp_max):
                    jlist.append(j-1)
                
        created = inp.copy()
        created = np.delete(created, ilist, 0)
        created = np.delete(created, jlist, 1)  

        alist.append(created)
        
    except:
        alist.append([])    # delete uinp_max rows&cols
           
    # .............................................................    
    for x in alist:
        
        if (len(x)!= 0):
            blist = f0000(x, [])
            for y in blist:
                clist.append(y) 
        else:
            for n in range(8):
                clist.append([])
     
    # .............................................................         
    return clist

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
version.append(40)
solvers.append(f0007)
    
# Example 14
inp, task, answer = get_arc(376,'train')
clist = f0007(inp, task)

for x in clist:
    if (np.array_equal(x, answer)):
        print('\nThis is correct:')
        plot_pic(x)  
        break

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def set_tasks_solvers(data1_path, data2_path, data_name): 
    
    with open(data1_path,'r') as f:
        tasks_name = list(json.load(f).keys())
    
    with open(data1_path,'r') as f:
        tasks_file = list(json.load(f).values())    
    
    with open(data2_path,'r') as f:
        tasks2_file = list(json.load(f).values())    
            
    # .................................................................................  
    print(f'{Style.BRIGHT}{Fore.CYAN}_' * 60)
    print('\n\t | Answers of solved tasks |', data_name, '|')
    print('_' * 60, '\n')   
    
    # .................................................................................  
    answer = []
    for n in range(len(tasks_file)):  
        task = tasks_file[n] 
   
        for sol in range(len(solvers)):
            for i in range(len(task['test'])):
                
                inp_test = np.array(task['test'][i]['input'])
                out_test = tasks2_file[n][i] 
                
                clist = solvers[sol](inp_test, task)
                for v in range(version[sol]):
                    
                    if (np.array_equal(clist[v], out_test)):
                        answer_ = clist[v].tolist()
                        
                        if (answer_ not in answer):
                            answer.append(answer_) 
                    
                        display(pd.DataFrame(data={'Answers for task': tasks_name[n],'Items': i,'Files': data_name},index=[n]))
                        display(pd.DataFrame(data={'The solver of this task': f'Function f000{sol}','version': v},index=[n]))
                    
                        plot_pic(answer_) 
                        print('\n', f'{Style.BRIGHT}{Fore.CYAN}|§|'*10, '✅✅✅', '|§|'*10, '\n')  
                        
                        break
                    
    # ............................................................................................................. 
    print(f'{Style.BRIGHT}{Fore.YELLOW}_' * 30)
    print('\nNumber of tasks solved :', len(answer))
    print(f'{Style.BRIGHT}{Fore.YELLOW}_' * 30)
    print(f'{Style.BRIGHT}{Fore.YELLOW}_' * 20)
    print(f'{Style.BRIGHT}{Fore.YELLOW}_' * 10)
    print()
    
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

set_tasks_solvers(train1_path, train2_path, 'Training File')

set_tasks_solvers(eval1_path, eval2_path, 'Evaluation File')  

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def set_submission(): 
    
    with open(sample_path,'r') as f:
        sample_sub = json.load(f) 
    
    with open(sample_path,'r') as f:
        sample_name = list(json.load(f).keys())
    
    # ................................................................................. 
    with open(test_path,'r') as f:
        tasks_name = list(json.load(f).keys())
    
    with open(test_path,'r') as f:
        tasks_file = list(json.load(f).values())    

    # .................................................................................  
    print(f'{Style.BRIGHT}{Fore.CYAN}_' * 60)
    print('\n\t | Answers of solved tasks |', 'Testing', '|')
    print('_' * 60, '\n')   
    
    # .................................................................................  
    answer = []
    for n in range(len(tasks_file)): 
        
        task = tasks_file[n] 
        len_train = len(task['train'])
        
        for sol in range(len(solvers)):
            counter = np.zeros(version[sol])
        
            for e in range(len_train):
                inp_e = np.array(task['train'][e]['input'])
                out_e = np.array(task['train'][e]['output'])
            
                try:
                    clist = solvers[sol](inp_e, task)
                    
                    for v in range(version[sol]): 
                        if (np.array_equal(clist[v], out_e)):
                            counter[v] += 1          
                except:
                    counter = counter       
            
            # .............................................................................................................
            for v in range(len(counter)):
                
                if (counter[v] == len_train):
            
                    for i in range(len(task['test'])):
                        inp_test = np.array(task['test'][i]['input'])
                        
                        clist = solvers[sol](inp_test, task)
                        answer_ = clist[v].tolist()
                        
                        sample_sub[tasks_name[n]][i]['attempt_1'] = answer_
                        if (answer_ not in answer):
                            answer.append(answer_) 
                        
                        display(pd.DataFrame(data={'Answers for task': tasks_name[n],'Items': i,'Attempt':'1','Files': 'Testing'},index=[n]))
                        print('\nNumber of correct answers for train_inputs of this task :', counter[v], '(All)')
                        display(pd.DataFrame(data={'The solver of this task': f'Function f000{sol}','version': v},index=[n]))
                        
                        plot_pic(answer_) 
                        print('\n', f'{Style.BRIGHT}{Fore.CYAN}|§|'*10, '✅✅✅', '|§|'*10, '\n')  
                        
                    break

                # .............................................................................................................  
                if (counter[v] == len_train-1):
                    
                    for i in range(len(task['test'])):
                        inp_test = np.array(task['test'][i]['input'])
                        
                        try:
                            clist = solvers[sol](inp_test, task)
                            answer_ = clist[v].tolist()
                            
                            sample_sub[tasks_name[n]][i]['attempt_2'] = answer_
                            if (answer_ not in answer):
                                answer.append(answer_) 
                            
                            display(pd.DataFrame(data={'Answers for task': tasks_name[n],'Items': i,'Attempt':'2','Files': 'Testing'},index=[n]))
                            print('\nNumber of correct answers for train_inputs of this task :', counter[v]) 
                            display(pd.DataFrame(data={'The solver of this task': f'Function f000{sol}','version': v},index=[n]))
                            
                            plot_pic(answer_) 
                            print('\n', f'{Style.BRIGHT}{Fore.YELLOW}|§|'*10, '☑️☑️☑️', '|§|'*10, '\n') 
                            
                        except:
                            break  
                    break 
                          
    # ............................................................................................................. 
    print(f'{Style.BRIGHT}{Fore.YELLOW}_' * 30)
    print('\nNumber of tasks solved :', len(answer))
    print(f'{Style.BRIGHT}{Fore.YELLOW}_' * 30)
    print(f'{Style.BRIGHT}{Fore.YELLOW}_' * 20)
    print(f'{Style.BRIGHT}{Fore.YELLOW}_' * 10)
    print()
    
    return sample_sub
    
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

sample_sub = set_submission()

with open('submission.json', 'w') as file:
    json.dump(sample_sub, file, indent=4)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 8  SelectZone & RotationFlip
def f0008(inp, task):
    '''
    This function returns the selected zone & RotationFlip.
    '''   
    clist = []
    alist = f0003(inp, task)
    
    for x in alist:
        try: 
            blist = f0000(x, [])
            for y in blist:
                clist.append(y)      
        except: 
            for i in range(8):
                clist.append([])
                
    # .............................................................    
    return clist

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: 
version.append(416)
solvers.append(f0008)           
    
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

sample_sub = set_submission()

with open('submission.json', 'w') as file:
    json.dump(sample_sub, file, indent=4)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


