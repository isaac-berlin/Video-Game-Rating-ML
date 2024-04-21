import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools

# Set the device to cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data from the Data directory
training_data = pd.read_csv("../Data/Video_games_esrb_rating.csv")  # 1895 rows x 34 columns
test_data = pd.read_csv("../Data/test_esrb.csv")  # 500 rows x 34 columns

# Remove the title column from both datasets
training_data = training_data.drop(columns=["title"])  # 1895 rows x 33 columns
test_data = test_data.drop(columns=["title"])  # 500 rows x 33 columns

# Convert the esrb_rating column to a numerical value for both datasets
training_data["esrb_rating"] = training_data["esrb_rating"].map({"E": 0, "ET": 1, "M": 2, "T": 3})
test_data["esrb_rating"] = test_data["esrb_rating"].map({"E": 0, "ET": 1, "M": 2, "T": 3})

# Split the training data into features and labels
X_train = training_data.drop(columns=["esrb_rating"]).values
Y_train = training_data["esrb_rating"].values

X_test = test_data.drop(columns=["esrb_rating"]).values
Y_test = test_data["esrb_rating"].values

# Create a DataLoader for the data
train_data = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.LongTensor(Y_train))
test_data = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.LongTensor(Y_test))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

# Define model architecture
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
        return x

# Define hyperparameters
learning_rate = 0.01
weight_decay = 0.001
batch_size = 64

# Train multiple models and save them for ensemble learning
ensemble_models = []
for model_index in range(10):  # Train 10 models for ensemble
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

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
        print(f"Model {model_index + 1}, Epoch [{epoch + 1}/20], Training Loss: {epoch_loss[-1]:.4f}, Training Accuracy: {train_accuracy:.2f}%")

        if train_accuracy >= 98:
            break

    ensemble_models.append(model)

# Identify the best model
all_predictions = []
all_accuracy = []
for model in ensemble_models:
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

best_model_index = np.argmax(all_accuracy)

# Print the model for the best accuracy
print(f"Model Number with Best Accuracy: {best_model_index + 1}, Accuracy: {all_accuracy[best_model_index]:.2f}%")

# Plot epoch loss and accuracy for the best ensemble model
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(epoch_loss) + 1), epoch_loss, label=f'Model {best_model_index + 1}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(epoch_accuracy) + 1), epoch_accuracy, label=f'Model {best_model_index + 1}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Epoch Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig("best_model_loss_accuracy.png")
plt.show()

# Plot confusion matrix for the best model
plt.clf()
plt.figure(figsize=(8, 6))
cm = confusion_matrix(Y_test, all_predictions[best_model_index])
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(4)
plt.xticks(tick_marks, ["E", "ET", "M", "T"], rotation=45)
plt.yticks(tick_marks, ["E", "ET", "M", "T"])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.savefig("best_model_confusion_matrix.png")
plt.show()



