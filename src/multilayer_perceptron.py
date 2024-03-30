import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

# set the device to cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the data from the Data directory
training_data = pd.read_csv("Data/Video_games_esrb_rating.csv") # 1895 rows x 34 columns
test_data = pd.read_csv("Data/test_esrb.csv") # 500 rows x 34 columns

# Remove the title column from both datasets
training_data = training_data.drop(columns=["title"]) # 1895 rows x 33 columns
test_data = test_data.drop(columns=["title"]) # 500 rows x 33 columns

# convert the esrb_rating column to a numerical value for both datasets
training_data["esrb_rating"] = training_data["esrb_rating"].map({"E": 0, "ET": 1, "M": 2, "T": 3})
test_data["esrb_rating"] = test_data["esrb_rating"].map({"E": 0, "ET": 1, "M": 2, "T": 3})

# split the training data into features and labels
X_train = training_data.drop(columns=["esrb_rating"]).values
Y_train = training_data["esrb_rating"].values

X_test = test_data.drop(columns=["esrb_rating"]).values
Y_test = test_data["esrb_rating"].values

# create a dataloader for the data
train_data = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

test_data = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

# define the model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(32, 64)
        self.linear2 = nn.Linear(64, 16)
        self.linear3 = nn.Linear(16, 4)

    def forward(self, x):
        out = self.linear1(x)
        out = torch.relu(out)
        
        out = self.linear2(out)
        out = torch.relu(out)
        
        out = self.linear3(out)
        
        return out
    

# create the model
model = MLP().to(device)

# define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train the model
num_epochs = 20

epoch_loss = []
epoch_accuracy = []


for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device).long()
        
        # forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)
        epoch_loss.append(loss.item())
        
        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    with torch.no_grad():
        correct = 0
        total = 0
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device).long()
            
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        epoch_accuracy.append(accuracy)
        print(f"Accuracy of the model on the test set: {accuracy:.2f}%")
        

# plot the loss and accuracy
plt.plot(epoch_loss)
plt.title("Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()
