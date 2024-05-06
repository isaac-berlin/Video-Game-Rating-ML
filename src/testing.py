import pandas as pd
import matplotlib.pyplot as plt

training_data = pd.read_csv("../Data/Video_games_esrb_rating.csv")
test_data = pd.read_csv("../Data/test_esrb.csv")

# Concatenate the training and test data
data = pd.concat([training_data, test_data], axis=0)

# Remove the title column
data = data.drop(columns=["title"])

# Map the esrb_rating column to numerical values
rating_map = {"E": 0, "ET": 1, "M": 2, "T": 3}
data["esrb_rating"] = data["esrb_rating"].map(rating_map)

# create a similarity matrix
similarity_matrix = data.corr()

# Plot the similarity matrix
plt.figure(figsize=(10, 10))
plt.matshow(similarity_matrix, fignum=1)
plt.xticks(range(similarity_matrix.shape[1]), similarity_matrix.columns, fontsize=10, rotation=90)
plt.yticks(range(similarity_matrix.shape[1]), similarity_matrix.columns, fontsize=10)
plt.colorbar()

plt.show()