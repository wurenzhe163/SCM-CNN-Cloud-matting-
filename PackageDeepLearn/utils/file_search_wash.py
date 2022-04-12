# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 19:20:10 2020

@author: 11733
"""
import os
import numpy as np

def delList(L):
    """"
    删除重复元素
    """
    L1 = []
    for i in L:
        if i not in L1:
            L1.append(i)
    return L1

# search_files = lambda path,endwith='.tif': [os.path.join(path,f) for f in os.listdir(path) if f.endswith(endwith) ]
def search_files(path,endwith='.tif'):
    """
    返回当前文件夹下文件
    Parameters
    ----------
    path : 路径
    endwith : The default is '.tif'.
    Returns: s ,   列表
    """
    s = []
    for f in os.listdir(path):
        if f.endswith(endwith):
            s.append(os.path.join(path,f))
    return s

def paixu(str_in,key=lambda info: int(info.split('_')[-1].split('.')[0])):
    test = sorted(str_in,key=key)
    return test



def search_files_alldir(path,endwith='.jpg',write=0):
    """
    遍历文件夹下所有文件（包括子文件夹），若只需当前文件夹下文件使用seach_files
    write = 0   返回矩阵
    write = 1   返回endwith[0:2].txt
    write = 2   返回endwith[0:2].txt，每行数据加引号
    """
    all_files = os.walk(path)#os.walk遍历所有文件，见P89
    s = []
    for i in all_files:
        for each_file in i[2]:
            if each_file.endswith(endwith):
                s.append(os.path.join(i[0],each_file))
    print('文件数=%d'%len(s))

    if write == 1 :
        with open(endwith[0:2]+'.txt','w') as f:
            for each in range(len(s)):
                if each+1<len(s):
                    f.write(s[each]+',')
                else:
                    f.write(s[each])
        print('writ is ok!')


    if write == 2 :
        with open(endwith[0:2]+'.txt','w') as f:
            for each in range(len(s)):
                if each+1<len(s):
                    f.write('"'+s[each]+'",')
                else:
                    f.write('"'+s[each]+'"')
        print('writ is ok!')
    return s


def filter_(img_array,label_array):
    """"
    根据label过滤img
    img_array-------->影像组
    label_array------>标签组
    """
    new_labelarray=[]
    new_imgarray=[]
    for i in range(len(label_array)):
        label_sum = np.sum(label_array[i])
        if label_sum !=0:
            new_labelarray.append(label_array[i])
            new_imgarray.append(img_array[i])
    return new_imgarray,new_labelarray
