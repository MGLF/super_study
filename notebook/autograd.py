from torch.autograd import Variable
import torch
import numpy
a=numpy.array([[2,3],[4,5]])
print(numpy.exp(a))
print(a)
x = Variable(torch.ones(2), requires_grad = True) #vairable是tensor的一个外包装
z=4*x*x
y=z.norm()
print(y)
