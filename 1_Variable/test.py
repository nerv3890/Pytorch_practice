import torch
from torch.autograd import Variable
import numpy as np

data = np.array([[1.0, 2.0], [3.0, 4.0]])
tensor1 = torch.from_numpy(data)
tensor2 = torch.FloatTensor([[1, 2], [3, 4]])

#print(
#    '\ndata:', data,
#    '\ntensor1', tensor1,
#    '\ntensor2', tensor2    
#        )

v = Variable(tensor1, requires_grad=True)

#print('tensor1: ', tensor1)
#print('v: ', v)

#t_out = torch.mean(tensor1*tensor1)       # x^2
#v_out = torch.mean(v*v)   # x^2
#print(t_out)
#print(v_out)

print(v)
print(v.data)
print(v.data.numpy())    
