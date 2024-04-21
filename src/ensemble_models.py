import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
# Load the dataset
file_path = 'Data\Video_games_esrb_rating.csv'
data = pd.read_csv(file_path)

# Displaying Useful Info: Distribution of ESRB_Rating
rating_counts = data['esrb_rating'].value_counts()
rating_percentages = (rating_counts / len(data)) * 100
print(rating_percentages)

# Display the first few rows of the dataframe
data.head()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Encode the 'esrb_rating' column
le = LabelEncoder()
data['esrb_rating_encoded'] = le.fit_transform(data['esrb_rating'])

# Split the dataset into features and target variable
X = data.drop(['title', 'esrb_rating', 'esrb_rating_encoded'], axis=1)
y = data['esrb_rating_encoded']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
random_forest = RandomForestClassifier(random_state=42)
bagging = BaggingClassifier(random_state=42)
boosting = GradientBoostingClassifier(random_state=42)
adaboostSAMME = AdaBoostClassifier(n_estimators=100, algorithm="SAMME",)
XGBoost = XGBClassifier(random_state=42)

# Train the models
random_forest.fit(X_train, y_train)
bagging.fit(X_train, y_train)
boosting.fit(X_train, y_train)
adaboostSAMME.fit(X_train, y_train)
XGBoost.fit(X_train, y_train)

# Predictions
rf_pred = random_forest.predict(X_test)
bagging_pred = bagging.predict(X_test)
boosting_pred = boosting.predict(X_test)
adaboostSAMME_pred = adaboostSAMME.predict(X_test)
XGBoost_pred = XGBoost.predict(X_test)

# Evaluate the models
rf_accuracy = accuracy_score(y_test, rf_pred)
bagging_accuracy = accuracy_score(y_test, bagging_pred)
boosting_accuracy = accuracy_score(y_test, boosting_pred)
adaboostSAMME_accuracy = accuracy_score(y_test, adaboostSAMME_pred)
XGBoost_accuracy = accuracy_score(y_test, XGBoost_pred)

rf_accuracy, bagging_accuracy, boosting_accuracy, adaboostSAMME_accuracy, XGBoost_accuracy
print("")
print("Random Forest Accuracy: ", rf_accuracy)
print("Random Forest Recall: ", recall_score(y_test, rf_pred, average='weighted'))
print("Random Forest Precision: ", precision_score(y_test, rf_pred, average='weighted'))
print("Random Forest F1 Score: ", f1_score(y_test, rf_pred, average='weighted'), "\n")

print("Bagging Accuracy: ", bagging_accuracy)
print("Bagging Recall: ", recall_score(y_test, bagging_pred, average='weighted'))
print("Bagging Precision: ", precision_score(y_test, bagging_pred, average='weighted'))
print("Bagging F1 Score: ", f1_score(y_test, bagging_pred, average='weighted'), "\n")

print("Boosting Accuracy: ", boosting_accuracy)
print("Boosting Recall: ", recall_score(y_test, boosting_pred, average='weighted'))
print("Boosting Precision: ", precision_score(y_test, boosting_pred, average='weighted'))
print("Boosting F1 Score: ", f1_score(y_test, boosting_pred, average='weighted'), "\n")

print("Adaptive Boosting SAMME Accuracy: ", adaboostSAMME_accuracy)
print("Adaptive Boosting SAMME Recall: ", recall_score(y_test, adaboostSAMME_pred, average='weighted'))
print("Adaptive Boosting SAMME Precision: ", precision_score(y_test, adaboostSAMME_pred, average='weighted'))
print("Adaptive Boosting SAMME F1 Score: ", f1_score(y_test, adaboostSAMME_pred, average='weighted'), "\n")

print("XGBoost Boosting Accuracy: ", XGBoost_accuracy)
print("XGBoost Boosting Recall: ", recall_score(y_test, XGBoost_pred, average='weighted'))
print("XGBoost Boosting Precision: ", precision_score(y_test, XGBoost_pred, average='weighted'))
print("XGBoost Boosting F1 Score: ", f1_score(y_test, XGBoost_pred, average='weighted'), "\n")

# Ensemble models on the test dataset


# Load the test dataset
test_file_path = r'Data\test_esrb.csv'
test_data = pd.read_csv(test_file_path)

# Display the first few rows of the dataframe
test_data.head()


# Encode the 'esrb_rating' column in the test dataset using the previously fitted LabelEncoder
test_data['esrb_rating_encoded'] = le.transform(test_data['esrb_rating'])

# Split the test dataset into features and target variable
X_test_set = test_data.drop(['title', 'esrb_rating', 'esrb_rating_encoded'], axis=1)
y_test_set = test_data['esrb_rating_encoded']

# Use the trained models to make predictions on the test set
rf_pred_test_set = random_forest.predict(X_test_set)
bagging_pred_test_set = bagging.predict(X_test_set)
boosting_pred_test_set = boosting.predict(X_test_set)
adaboostSAMME_pred_test_set = adaboostSAMME.predict(X_test_set)
XGBoost_pred_test_set = XGBoost.predict(X_test_set)

# Evaluate the models on the test set
rf_accuracy_test_set = accuracy_score(y_test_set, rf_pred_test_set)
bagging_accuracy_test_set = accuracy_score(y_test_set, bagging_pred_test_set)
boosting_accuracy_test_set = accuracy_score(y_test_set, boosting_pred_test_set)
adaboostSAMME_accuracy_test_set = accuracy_score(y_test_set, adaboostSAMME_pred_test_set)
XGBoost_accuracy_test_set = accuracy_score(y_test_set, XGBoost_pred_test_set)

# Performance metrics
rf_recall_test_set = recall_score(y_test_set, rf_pred_test_set, average='weighted')
rf_precision_test_set = precision_score(y_test_set, rf_pred_test_set, average='weighted')
rf_f1_test_set = f1_score(y_test_set, rf_pred_test_set, average='weighted')

bagging_recall_test_set = recall_score(y_test_set, bagging_pred_test_set, average='weighted')
bagging_precision_test_set = precision_score(y_test_set, bagging_pred_test_set, average='weighted')
bagging_f1_test_set = f1_score(y_test_set, bagging_pred_test_set, average='weighted')

boosting_recall_test_set = recall_score(y_test_set, boosting_pred_test_set, average='weighted')
boosting_precision_test_set = precision_score(y_test_set, boosting_pred_test_set, average='weighted')
boosting_f1_test_set = f1_score(y_test_set, boosting_pred_test_set, average='weighted')

adaboostSAMME_recall_test_set = recall_score(y_test_set, adaboostSAMME_pred_test_set, average='weighted')
adaboostSAMME_precision_test_set = precision_score(y_test_set, adaboostSAMME_pred_test_set, average='weighted')
adaboostSAMME_f1_test_set = f1_score(y_test_set, adaboostSAMME_pred_test_set, average='weighted')

XGBoost_recall_test_set = recall_score(y_test_set, XGBoost_pred_test_set, average='weighted')
XGBoost_precision_test_set = precision_score(y_test_set, XGBoost_pred_test_set, average='weighted')
XGBoost_f1_test_set = f1_score(y_test_set, XGBoost_pred_test_set, average='weighted')

# Display the results
results = {
    "Model": ["Random Forest", "Bagging", "Boosting", "AdaBoostSAMME", "XGBoost"],
    "Accuracy": [rf_accuracy_test_set, bagging_accuracy_test_set, boosting_accuracy_test_set, adaboostSAMME_accuracy_test_set, XGBoost_accuracy_test_set],
    "Recall": [rf_recall_test_set, bagging_recall_test_set, boosting_recall_test_set, adaboostSAMME_recall_test_set, XGBoost_recall_test_set],
    "Precision": [rf_precision_test_set, bagging_precision_test_set, boosting_precision_test_set, adaboostSAMME_precision_test_set, XGBoost_precision_test_set],
    "F1 Score": [rf_f1_test_set, bagging_f1_test_set, boosting_f1_test_set, adaboostSAMME_f1_test_set, XGBoost_f1_test_set]
}

print(pd.DataFrame(results))


# Plotting the accuracy of the models
models = ["Random Forest", "Bagging", "Boosting", "AdaBoostSAMME", "XGBoost"]

validation_accuracies_actual = [0.8522, 0.8549, 0.8443, 0.7599, 0.8628]
test_accuracies_actual = [0.9103, 0.9082, 0.8786, 0.7604, 0.9108]

bar_width = 0.35
index = np.arange(len(models))

fig, ax = plt.subplots()
bar1 = ax.bar(index, validation_accuracies_actual, bar_width, label='Validation Accuracy')
bar2 = ax.bar(index + bar_width, test_accuracies_actual, bar_width, label='Test Accuracy')

ax.set_xlabel('Model')
ax.set_ylabel('Accuracy')
ax.set_title('Validation vs Test Accuracy for Different Models')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(models)
ax.legend()

plt.show()
def plot_confusion_matrix(y_test, predictions, title='Confusion Matrix', filename='confusion_matrix.png'):
    plt.clf()
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, predictions)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(4)
    plt.xticks(tick_marks, ["E", "ET", "M", "T"], rotation=45)
    plt.yticks(tick_marks, ["E", "ET", "M", "T"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.savefig(filename)
    plt.show()

plot_confusion_matrix(y_test_set, rf_pred_test_set, 'Confusion Matrix for RF', 'rf_confusion_matrix.png')
plot_confusion_matrix(y_test_set, bagging_pred_test_set, 'Confusion Matrix for Bagging', 'bagging_confusion_matrix.png')
plot_confusion_matrix(y_test_set, boosting_pred_test_set, 'Confusion Matrix for Boosting', 'boosting_confusion_matrix.png')
plot_confusion_matrix(y_test_set, adaboostSAMME_pred_test_set, 'Confusion Matrix for AdaBoost', 'adaboost_confusion_matrix.png')
plot_confusion_matrix(y_test_set, XGBoost_pred_test_set, 'Confusion Matrix for XGBoost', 'xgboost_confusion_matrix.png')

#balance dataset


random_forest_balanced = RandomForestClassifier(random_state=42, class_weight='balanced')



# Train the models
random_forest_balanced.fit(X_train, y_train)

# Predictions
rf_pred_balanced = random_forest_balanced.predict(X_test)



# Evaluate the models
rf_accuracy_balanced = accuracy_score(y_test, rf_pred_balanced)


print("Random Forest Accuracy (Balanced): ", rf_accuracy_balanced)
print("Random Forest Recall (Balanced): ", recall_score(y_test, rf_pred_balanced, average='weighted'))
print("Random Forest Precision (Balanced): ", precision_score(y_test, rf_pred_balanced, average='weighted'))
print("Random Forest F1 Score (Balanced): ", f1_score(y_test, rf_pred_balanced, average='weighted'), "\n")



# Use the trained models to make predictions on the test set
rf_pred_test_set_balanced = random_forest_balanced.predict(X_test_set)


# Evaluate the models on the test set
rf_accuracy_test_set_balanced = accuracy_score(y_test_set, rf_pred_test_set_balanced)


# Performance metrics
rf_recall_test_set_balanced = recall_score(y_test_set, rf_pred_test_set_balanced, average='weighted')
rf_precision_test_set_balanced = precision_score(y_test_set, rf_pred_test_set_balanced, average='weighted')
rf_f1_test_set_balanced = f1_score(y_test_set, rf_pred_test_set_balanced, average='weighted')


# Display the results
results_balanced = {
    "Model": ["Random Forest (Balanced)"],
    "Accuracy": [rf_accuracy_test_set_balanced],
    "Recall": [rf_recall_test_set_balanced],
    "Precision": [rf_precision_test_set_balanced],
    "F1 Score": [rf_f1_test_set_balanced]
}

print(pd.DataFrame(results_balanced))

plot_confusion_matrix(y_test_set, rf_pred_test_set_balanced, 'Confusion Matrix for RF Balanced', 'rf_balanced_confusion_matrix.png')
