from numpy import array, dot
from qpsolvers import solve_qp

import time
M = array([[1., 2., 0.], [-8., 3., 2.], [0., 1., 1.]])
P = dot(M.T, M)  # quick way to build a symmetric matrix
q = dot(array([3., 2., 3.]), M).reshape((3,))
G = array([[1., 2., 1.], [2., 0., 1.], [-1., 2., -1.]])
h = array([3., 2., -2.]).reshape((3,))

t_start = time.time()
print "QP solution:", solve_qp(P, q, G, h)
t_end = (time.time())
print t_end-t_start
