import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import itertools

# Set the device to cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data
training_data = pd.read_csv("../Data/Video_games_esrb_rating.csv")
test_data = pd.read_csv("../Data/test_esrb.csv")

# Remove the title column
training_data = training_data.drop(columns=["title"])
test_data = test_data.drop(columns=["title"])

# Map the esrb_rating column to numerical values
rating_map = {"E": 0, "ET": 1, "M": 2, "T": 3}
training_data["esrb_rating"] = training_data["esrb_rating"].map(rating_map)
test_data["esrb_rating"] = test_data["esrb_rating"].map(rating_map)

# Prepare features and labels
X_train = training_data.drop(columns=["esrb_rating"]).values
Y_train = training_data["esrb_rating"].values
X_test = test_data.drop(columns=["esrb_rating"]).values
Y_test = test_data["esrb_rating"].values

# Create data loaders
train_data = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.LongTensor(Y_train))
test_data = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.LongTensor(Y_test))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

# Define the model architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc_layers = nn.ModuleList([
            nn.Linear(32, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 4)
        ])
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        for layer in self.fc_layers[:-1]:
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.fc_layers[-1](x)
        x = F.softmax(x, dim=1)
        return x

# Training multiple models
models = []
models_loss = []
models_accuracy = []
for model_index in range(10):
    model = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()
    epoch_loss = []
    epoch_accuracy = []
    for epoch in range(20):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_accuracy = 100 * correct / total
        epoch_loss.append(running_loss / len(train_loader))
        epoch_accuracy.append(train_accuracy)
        print(f"Model {model_index + 1}, Epoch [{epoch + 1}/20], Loss: {epoch_loss[-1]:.4f}, Accuracy: {train_accuracy:.2f}%")
        if train_accuracy >= 98:
            break
    models.append(model)
    models_loss.append(epoch_loss)
    models_accuracy.append(epoch_accuracy)

# Evaluating models
all_predictions = []
all_accuracy = []
for model in models:
    model.eval()
    predictions = []
    with torch.no_grad():
        for features, _ in test_loader:
            features = features.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
    all_predictions.append(predictions)
    correct = np.sum(np.array(predictions) == Y_test)
    total = len(Y_test)
    all_accuracy.append(correct / total * 100)

# Find the best model
best_model_index = np.argmax(all_accuracy)
print(f"Best Model: Model {best_model_index + 1}, Accuracy: {all_accuracy[best_model_index]:.2f}%")

# Classification report for the best model
best_predictions = all_predictions[best_model_index]
print("Classification Report for the Best Model:")
print(classification_report(Y_test, best_predictions, target_names=["E", "ET", "M", "T"]))

# Plotting confusion matrix, loss and accuracy for the best model
plt.figure(figsize=(10, 8))
cm = confusion_matrix(Y_test, best_predictions)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(4)
plt.xticks(tick_marks, ["E", "ET", "M", "T"], rotation=45)
plt.yticks(tick_marks, ["E", "ET", "M", "T"])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.show()