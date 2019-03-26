import torch
import matplotlib.pyplot as plt

a = torch.linspace(-1, 1, 100) # shape=(1,100)
# Generate the dataset
#print(a, a.size())
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)# x data (tensor), shape=(100, 1)
#print(x, x.size())
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

#plt.scatter(x.data.numpy(), y.data.numpy())
#plt.show()

import torch
import torch.nn.functional as F     

# Build the net
class Net(torch.nn.Module): 
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   
        self.predict = torch.nn.Linear(n_hidden, n_output) 

    def forward(self, x):   
        x = F.relu(self.hidden(x))      
        x = self.predict(x)       
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)

print(net)  # net 的结构
"""
Net (
  (hidden): Linear (1 -> 10)
  (predict): Linear (10 -> 1)
)
"""

# Train the net
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  
loss_func = torch.nn.MSELoss() 

for t in range(100):
    prediction = net(x)    

    loss = loss_func(prediction, y)    

    optimizer.zero_grad()   
    loss.backward()         
    optimizer.step()

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1) 
