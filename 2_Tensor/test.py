import torch
import numpy as np

np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()
print(
    '\nnumpy array:', np_data,          # [[0 1 2], [3 4 5]]
    '\ntorch tensor:', torch_data,      #  0  1  2 \n 3  4  5    [torch.LongTensor of size 2x3]
    '\ntensor to array:', tensor2array, # [[0 1 2], [3 4 5]]
)

#np_data2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
np_data2 = np.arange(1,9,1).reshape((3, 3))
torch_data2 = torch.from_numpy(np_data2)
tensor2array2 = torch_data2.numpy()

print(
    '\nnumpy array:', np_data2,          # [[0 1 2], [3 4 5]]
    '\ntorch tensor:', torch_data2,      #  0  1  2 \n 3  4  5    [torch.LongTensor of size 2x3]
    '\ntensor to array:', tensor2array2, # [[0 1 2], [3 4 5]]
)
