import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Set device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
data = pd.read_csv("../Data/Video_games_esrb_rating.csv")  # Assuming it's the same dataset
data = data.drop(columns=["title"])  # Remove title column

# Convert labels to numerical values
data["esrb_rating"] = data["esrb_rating"].map({"E": 0, "ET": 1, "M": 2, "T": 3})

# Split data into features and labels
X = data.drop(columns=["esrb_rating"]).values
Y = data["esrb_rating"].values

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert numpy arrays to PyTorch tensors and move to device
X_train = torch.Tensor(X_train).to(device)
X_test = torch.Tensor(X_test).to(device)
Y_train = torch.LongTensor(Y_train).to(device)
Y_test = torch.LongTensor(Y_test).to(device)

# Define MLP model with changes in architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(32, 512)   # Input size 32 (features), output size 512
        self.fc2 = nn.Linear(512, 512)  # Increased neurons in hidden layer
        self.fc3 = nn.Linear(512, 256)  # Adjusted layer size
        self.fc4 = nn.Linear(256, 128)  # Adjusted layer size
        self.fc5 = nn.Linear(128, 64)   # Adjusted layer size
        self.fc6 = nn.Linear(64, 4)     # Output layer with 4 classes (E, ET, M, T)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # ReLU activation for all hidden layers
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)          # No activation for output layer
        return x

# Initialize model, loss, optimizer
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for multi-class classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)  # Reduce learning rate on plateau

# Training loop with increased epochs
num_epochs = 100  # Increased number of epochs for training
best_accuracy = 0
early_stopping_counter = 0
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    # Train
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
    # Validation
    model.eval()
    with torch.no_grad():
        train_accuracy = torch.sum(torch.argmax(outputs, dim=1) == Y_train).item() / len(Y_train)
        train_accuracies.append(train_accuracy)
        
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, Y_test)
        test_losses.append(test_loss.item())
        
        test_accuracy = torch.sum(torch.argmax(test_outputs, dim=1) == Y_test).item() / len(Y_test)
        test_accuracies.append(test_accuracy)
        
        # Learning rate scheduler
        scheduler.step(test_loss)
        
        # Early stopping
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= 10:
                print(f'Early stopping at epoch {epoch}, best accuracy: {best_accuracy}')
                break

# Evaluate on test set and provide debugging information
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    predictions = torch.argmax(test_outputs, dim=1)
    accuracy = torch.sum(predictions == Y_test).item() / len(Y_test)
    print(f'Test accuracy: {accuracy:.2f}')

    # Confusion matrix
    print('Confusion Matrix:')
    print(confusion_matrix(Y_test.cpu(), predictions.cpu()))

    # Classification report
    print('Classification Report:')
    print(classification_report(Y_test.cpu(), predictions.cpu(), target_names=["E", "ET", "M", "T"]))

# Plotting
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(test_losses, label='Test')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train')
plt.plot(test_accuracies, label='Test')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
