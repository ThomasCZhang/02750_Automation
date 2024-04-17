import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MyModel(nn.Module):
    def __init__(self, in_feat:int, out_feat:int):
        super().__init__()
        self.fc_in = nn.Linear(in_feat, 16, dtype=torch.float32)
        self.fc1 = nn.Linear(16, 64, dtype=torch.float32)
        # self.fc2 = nn.Linear(64, 256, dtype=torch.float32)
        # self.fc3 = nn.Linear(256, 64, dtype=torch.float32)
        self.fc4 = nn.Linear(64, 16, dtype=torch.float32)
        self.fc_out = nn.Linear(16, out_feat, dtype=torch.float32)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p = 0.05)

        self.device = 'cpu'
        # if torch.cuda.is_available():
        #     self.device = 'cuda'
        # elif torch.backends.mps.is_available():
        #     self.device = 'mps'
        
        self.to(self.device)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype = torch.float32, requires_grad = True, device=self.device)
        
        x = self.fc_in(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # x = self.fc2(x)
        # x = self.activation(x)
        # x = self.dropout(x)

        # x = self.fc3(x)
        # x = self.activation(x)
        # x = self.dropout(x)

        x = self.fc4(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fc_out(x)
        x = F.softmax(x, dim = 1)
        return x
    
    def fit(self, x, y, verbose: bool=False):
        if isinstance(x, np.ndarray):
            raw_x = torch.tensor(x, dtype = torch.float32, requires_grad = True, device=self.device)
        if isinstance(y, np.ndarray):
            raw_y = torch.tensor(y, dtype=torch.long, device = self.device)
        self.reset_weights()

        num_epochs = 100
        my_data = CustomDataset(raw_x, raw_y)
        my_loader = DataLoader(my_data, batch_size = 3, shuffle = True)

        loss_fn = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(self.parameters(), lr = 0.03, momentum=0.3)
        optimizer = optim.Adam(self.parameters(), lr = 0.002)

        for i in range(num_epochs):
            self.train()
            for j, data in enumerate(my_loader):
                x, y = data
                optimizer.zero_grad()
                outputs = self(x)
                loss = loss_fn(outputs, y)
                loss.backward()
                optimizer.step()

            if verbose:
                if i % 10 == 0:
                    self.eval()
                    loss = 0
                    outputs = self(raw_x)
                    loss = loss_fn(outputs, raw_y)
                    print(loss.item())

    def predict_proba(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype = torch.float32, device = self.device)
        with torch.no_grad():
            self.eval()
            return self.forward(x).detach().cpu().numpy()

    def predict(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype = torch.float32, device = self.device)
        with torch.no_grad():
            self.eval()
            pred_y = self.forward(x).detach().cpu()
            return torch.argmax(pred_y, dim = 1).numpy()
        
    def reset_weights(self):
        for name, child in self.named_children():
            if 'fc' in name:
                child.reset_parameters()
        # self.fc_in.reset_parameters()
        # self.fc1.reset_parameters()
        # self.fc4.reset_parameters()
        # self.fc_out.reset_parameters()

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
if __name__ == '__main__':
    # Use the same data as question 2.
    filepath = "/Users/thomaszhang/Development/Python/02750_Automation_2024/hw4/hw4_ex2_data.csv"
    data = []
    with open(filepath) as f:
        header = f.readline().strip().split(',')
        print('Header Categories:', end = " ")
        for i, category in enumerate(header): print(f"col {i}: {category}    ", end = "")
        for line in f:
            line = line.strip().split(',')
            line = [float(x) for x in line]
            data.append(line)
    data = np.array(data)
    x = data[:, :2]
    y = data[:, 2]
    print(f'\nThe shape of the data is: {data.shape}\nFeature shape: {x.shape}\nLabel shape: {y.shape}')
    print(f'Number of unique classes {len(np.unique(y))}')

    x, y = torch.tensor(x, dtype = torch.float64, requires_grad=True), torch.tensor(y, dtype = torch.long)
    clf = MyModel(x.shape[1], 3)
    clf.fit(x, y)
    pred_y = clf.predict(x)
    print(f'Accuracy: {np.sum(pred_y == y.numpy())/len(y)}')