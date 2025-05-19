import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(60, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 40)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, file_path):
        data = np.loadtxt(file_path)
        self.X = torch.FloatTensor(data[:, :60])
        self.y = torch.FloatTensor(data[:, 60:])

        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = CustomDataset("Dataset10000.txt")
train_dataset = torch.utils.data.Subset(dataset, range(9950))
dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)  


def train_model(epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_dataset):.4f}")

if __name__ == "__main__":
    model = NeuralNetwork()
    criterion = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start = time.time()
    train_model(epochs=100)
    print(start - time.time())

    # test of training
    datatest = []
    for i in range(50):
        x = dataset.X[-(i+1)]
        y = dataset.y[-(i+1)]

        y_pred = model(x)
        loss = torch.mean(torch.abs(y_pred-y))
        datatest.append(loss)
#    print(torch.mean(torch.tensor(datatest)))

    # Results
    x = dataset.X[-8]     #type any data
    y = dataset.y[-8]

    y_pred = model(x)

    print(f"y real: {y}")
    print(f"y pred: {y_pred}")
    print(f"mistake: {torch.mean(torch.abs(y_pred-y))}")