# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 22:02:13 2022

@author: A
"""


import sys
 
def func(a,b):
    return (a+b)
 
if __name__ == '__main__':
    a = []
    for i in range(1, len(sys.argv)):
        a.append((int(sys.argv[i])))
 
    print(func(a[0],a[1]))
#    print("11")