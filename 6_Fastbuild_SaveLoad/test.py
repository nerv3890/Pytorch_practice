import torch 
import matplotlib.pyplot as plt

# Make data
n_data = torch.ones(100, 2)
#print(n_data)

x0 = torch.normal(2*n_data, 1) 
y0 = torch.zeros(100)

x1 = torch.normal(-2*n_data, 1)    
y1 = torch.ones(100)           

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor) # The default data type of label in Pytorch 

plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')

#plt.scatter(x.data.numpy(), y.data.numpy())
#plt.show()

import torch.nn.functional as F     

# Method 1
# Build the net (Standard procedure of Pytorch)
class Net(torch.nn.Module): 
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   
        self.predict = torch.nn.Linear(n_hidden, n_output) 

    def forward(self, x):   
        x = F.relu(self.hidden(x))      
        x = self.predict(x)       
        return x

net1 = Net(n_feature=2, n_hidden=10, n_output=2)

# Method 2 Fast build 
net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),    
    torch.nn.Linear(10, 2),
)

print(net1) 
print(net2)
"""
Net (
  (hidden): Linear (1 -> 10)
  (predict): Linear (10 -> 1)
)
"""

# Train the net
optimizer = torch.optim.SGD(net2.parameters(), lr=0.01)  
loss_func = torch.nn.CrossEntropyLoss() 

for epoch in range(100):
    out = net2(x)    

    loss = loss_func(out, y) # F.softmax(out)   

    optimizer.zero_grad() # Reset the gradient before backward computing   
    loss.backward()        
    optimizer.step() # Give the optimizier the new weights

    if epoch % 2 == 0:
        plt.cla()
        prediction = torch.max(F.softmax(out), 1)[1] # Return the max value of out in diemtion 1
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlBu')
        accuracy = sum(pred_y == target_y)/200.  
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
        if epoch % 10 == 0:
            plt.savefig('epoch_'+str(epoch)+'.png')

torch.save(net2, 'net2.pkl') # Entire net
torch.save(net2.state_dict(), 'net2_params.pkl')

def restore_net():
    # restore entire net1 to net2
    net2 = torch.load('net2.pkl')
    prediction = net2(x)

def restore_params():
    net2.load_state_dict(torch.load('net2_params.pkl'))
    prediction = net2(x)
