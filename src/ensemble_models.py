import pandas as pd

# Load the dataset
file_path = r'Video-Game-Rating-ML\Data\Video_games_esrb_rating.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe
data.head()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

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
gradient = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=5, random_state=42)

# Train the models
random_forest.fit(X_train, y_train)
bagging.fit(X_train, y_train)
boosting.fit(X_train, y_train)
adaboostSAMME.fit(X_train, y_train)
gradient.fit(X_train, y_train)

# Predictions
rf_pred = random_forest.predict(X_test)
bagging_pred = bagging.predict(X_test)
boosting_pred = boosting.predict(X_test)
adaboostSAMME_pred = adaboostSAMME.predict(X_test)
gradient_pred = gradient.predict(X_test)

# Evaluate the models
rf_accuracy = accuracy_score(y_test, rf_pred)
bagging_accuracy = accuracy_score(y_test, bagging_pred)
boosting_accuracy = accuracy_score(y_test, boosting_pred)
adaboostSAMME_accuracy = accuracy_score(y_test, adaboostSAMME_pred)
gradient_accuracy = accuracy_score(y_test, gradient_pred)

rf_accuracy, bagging_accuracy, boosting_accuracy, adaboostSAMME_accuracy, gradient_accuracy
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

print("Gradient Boosting Accuracy: ", gradient_accuracy)
print("Gradient Boosting Recall: ", recall_score(y_test, gradient_pred, average='weighted'))
print("Gradient Boosting Precision: ", precision_score(y_test, gradient_pred, average='weighted'))
print("Gradient Boosting F1 Score: ", f1_score(y_test, gradient_pred, average='weighted'), "\n")

# Ensemble models on the test dataset


# Load the test dataset
test_file_path = r'Video-Game-Rating-ML\Data\test_esrb.csv'
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
gradient_pred_test_set = gradient.predict(X_test_set)

# Evaluate the models on the test set
rf_accuracy_test_set = accuracy_score(y_test_set, rf_pred_test_set)
bagging_accuracy_test_set = accuracy_score(y_test_set, bagging_pred_test_set)
boosting_accuracy_test_set = accuracy_score(y_test_set, boosting_pred_test_set)
adaboostSAMME_accuracy_test_set = accuracy_score(y_test_set, adaboostSAMME_pred_test_set)
gradient_accuracy_test_set = accuracy_score(y_test_set, gradient_pred_test_set)

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

gradient_recall_test_set = recall_score(y_test_set, gradient_pred_test_set, average='weighted')
gradient_precision_test_set = precision_score(y_test_set, gradient_pred_test_set, average='weighted')
gradient_f1_test_set = f1_score(y_test_set, gradient_pred_test_set, average='weighted')

# Display the results
results = {
    "Model": ["Random Forest", "Bagging", "Boosting", "AdaBoostSAMME", "Gradient Boosting"],
    "Accuracy": [rf_accuracy_test_set, bagging_accuracy_test_set, boosting_accuracy_test_set, adaboostSAMME_accuracy_test_set, gradient_accuracy_test_set],
    "Recall": [rf_recall_test_set, bagging_recall_test_set, boosting_recall_test_set, adaboostSAMME_recall_test_set, gradient_recall_test_set],
    "Precision": [rf_precision_test_set, bagging_precision_test_set, boosting_precision_test_set, adaboostSAMME_precision_test_set, gradient_precision_test_set],
    "F1 Score": [rf_f1_test_set, bagging_f1_test_set, boosting_f1_test_set, adaboostSAMME_f1_test_set, gradient_f1_test_set]
}

print(pd.DataFrame(results))
