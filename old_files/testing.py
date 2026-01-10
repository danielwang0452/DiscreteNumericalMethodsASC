import torch
import torch.nn as nn
import torch.nn.functional as F

# setup
torch.manual_seed(0)
n = 3
k = 2

pi = F.softmax(torch.randn(n)) # n
grad = torch.randn((n, n)) # n, n
I = torch.eye(n)

def delta_ij(i, j):
    if i == j:
        return 1
    else:
        return 0

'''
LHS = 0
for j in range(n):
    LHS += (pi[j] / 2) * (pi[k]) * (grad[j] @ I[:, j])

LHS_2 = 0
for i in range(n):
    LHS_2 += (pi[k]-pi[i]-1)*I[:, i]/2
LHS += pi[k] * grad[k] @ (I[k] - LHS_2)
'''
# 1. eq 4

eq4 = 0
RHS_sum1 = 0
RHS_sum2 = 0
for j in range(n):
    RHS_sum1 += pi[j]*I[j]
    RHS_sum2 += (pi[k]*pi[j]/2)*(grad[j]@(I[k]-I[j]))
eq4 += (pi[k] / 2) * (grad[k] @ (I[k] - RHS_sum1)) + RHS_sum2

# 2. E[Reinmax]
Ereinmax = torch.zeros((n))
for j in range(n):
    pid = (pi+I[:, j])/2
    Ereinmax += pi[j]*(2*grad[j]@(torch.outer(pid, torch.ones((n)))*I - torch.outer(pid, pid))-0.5*grad[j]@(torch.outer(pi, torch.ones((n)))*I - torch.outer(pi, pi)))
    print(j, torch.outer(pid, torch.ones((n)))*I - torch.outer(pid, pid))
# 3. Eq 7 scalar
eq7 = 0

for j in range(n):
    pid = (pi + I[j]) / 2
    inner_sum_1 = torch.zeros(n)
    inner_sum_2 = torch.zeros(n)
    for i in range(n):
        inner_sum_1 += (pi[i]+delta_ij(i, j))*0.5*I[i]
        inner_sum_2 += pi[i]*I[i]
    print(inner_sum_1, pid)
    vector = (pi[k]+delta_ij(k, j))*(I[k]-inner_sum_1)-(0.5*pi[k])*(I[k]-inner_sum_2)
    eq7 += pi[j]* (grad[j] @ vector)
    #print(j, 0.5 * (pi[k] + delta_ij(k, j)) * (I[k] - inner_sum_1))

# 4. eq 7 vectorised
eq7_v = 0
for j in range(n):
    pid = (pi + I[j]) / 2
    vector = 2*pid[k]*(I[k]-pid)-(0.5*pi[k])*(I[k]-pi)
    eq7_v += pi[j] * (grad[j] @ vector)
    #print(j, 0.5 * (pi[k] + delta_ij(k, j)) * (I[k] - inner_sum_1))


print(Ereinmax, eq7_v, eq4, eq7)

