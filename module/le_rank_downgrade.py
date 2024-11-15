#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
'''
对于方程组 Ax = b ，若存在可逆矩阵 P,Q ，使得：

# QAP = [A11 A12]
#       [ 0  A22]

若解出 QAPy = Qb 则 Py = x ，特别的，若记

# y = [y1 y2]^T
# Qb= [c1 c2]^T

则
# [A11 A12][y1] = [c1]
# [ 0  A22][y2]   [c2]

# A22y2 = c2
# A11y1 = c1 - A12y2
方程组变为两个低阶方程组。
'''

def downgrade(A, b, solver):
    pass