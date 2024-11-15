#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from numbers import Number

__all__ = ["swap", "swaprow", "swapcol", "arrange", "arrangerow", "arrangecol",
           "mul_perrow", "matmul"]

def matmul(a:np.ndarray, *b:np.ndarray) -> np.ndarray:
    result = a.copy()
    for bi in b:
        result = np.matmul(result, bi)
    return result

def swap(a:np.ndarray, axis:int, index0:int, index1:int) -> None:
    indexing = [":",]*(len(a.shape))
    indexing[axis] = str(index0)
    i0 = ",".join(indexing)
    indexing[axis] = str(index1)
    i1 = ",".join(indexing)
    del indexing
    exec(f"a[{i0}],a[{i1}]=a[{i1}].copy(),a[{i0}].copy()")

def swaprow(a:np.ndarray, row0:int, row1:int) -> None:
    a[row0,:],a[row1,:] = a[row1,:].copy(),a[row0,:].copy()

def swapcol(a:np.ndarray, col0:int, col1:int) -> None:
    a[:,col0],a[:,col1] = a[:,col1].copy(),a[:,col0].copy()

def arrange(a:np.ndarray, axis:int, indexes=list[int]) -> None:
    if (shape:=a.shape)[axis] != len(indexes):
        raise ValueError("List of indexes must have the same length with the array axis.")
    b = np.zeros(shape)
    #
    class i:
        indexing = [":",]*(len(shape))
        def __getitem__(self, index):
            i.indexing = str(index)
            return ",".join(i.indexing)
    #
    for index in range(len(indexes)):
        exec(f"b[{i[index]}]=a[{i[indexes[index]]}]")
    a[:]=b[:]

def arrangerow(a:np.ndarray, indexes=list[int]) -> None:
    """PA"""
    if (shape:=a.shape)[0] != len(indexes):
        raise ValueError("List of indexes must have the same length with the array row.")
    b = np.zeros(shape)
    for index in range(len(indexes)):
        b[index,:] = a[indexes[index],:]
    a[:]=b[:]

def arrangerow_undo(a:np.ndarray, indexes=list[int]) -> None:
    """P^T A or P^{-1} A"""
    if (shape:=a.shape)[0] != len(indexes):
        raise ValueError("List of indexes must have the same length with the array row.")
    b = np.zeros(shape)
    for index in range(len(indexes)):
        b[indexes[index],:] = a[index,:]
    a[:]=b[:]

def arrangecol(a:np.ndarray, indexes=list[int]) -> None:
    """AP"""
    if (shape:=a.shape)[1] != len(indexes):
        raise ValueError("List of indexes must have the same length with the array column.")
    b = np.zeros(shape)
    for index in range(len(indexes)):
        b[:,index] = a[:,indexes[index]]
    a[:]=b[:]

def arrangecol_undo(a:np.ndarray, indexes=list[int]) -> None:
    """AP^T or AP^{-1}"""
    if (shape:=a.shape)[1] != len(indexes):
        raise ValueError("List of indexes must have the same length with the array column.")
    b = np.zeros(shape)
    for index in range(len(indexes)):
        b[:,indexes[index]] = a[:,index]
    a[:]=b[:]

def mul_perrow(a:np.ndarray, others=list[Number]) -> np.ndarray:
    if (shape:=a.shape[0]) != len(others):
        raise ValueError("List of others must have the same length with the array row.")
    for i in range(shape):
        a[i,:]*=others[i]
    return a
    