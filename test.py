import torch
import torch.utils.data as Data
torch.manual_seed(1)    # reproducible

BATCH_SIZE = 5    

x = torch.linspace(1, 10, 10)       # x data (torch tensor)
y = torch.linspace(10, 1, 10)       # y data (torch tensor)

# Transform into the data type that Pytorch can accept
torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)

loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               
    num_workers=2,         
)

for epoch in range(3): 
    for step, (batch_x, batch_y) in enumerate(loader): 
        
        # Here is the traing step...       
        
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())
