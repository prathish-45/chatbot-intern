import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

# Assuming X_train, y_train, X_test, y_test, and tag_list are already defined
# You need to define or load these variables before running this script

# Example placeholder data (replace with actual data)
X_train = np.random.rand(100, 10)  # 100 samples, 10 features
y_train = np.random.randint(0, 2, 100)  # 100 samples, binary classification
X_test = np.random.rand(20, 10)  # 20 samples, 10 features
y_test = np.random.randint(0, 2, 20)  # 20 samples, binary classification
tag_list = ['class_0', 'class_1']  # Example tag list for binary classification

# Train Logistic Regression Model
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'model.pkl')

# Save the tag list
joblib.dump(tag_list, 'tag_list.pkl')

# Evaluate Model
print("Model Performance:")
y_pred = model.predict(X_test)  # Predict the labels for X_test
unique_labels = np.unique(y_test)  # Get unique classes in y_test
filtered_tag_list = [tag_list[label] for label in unique_labels]  # Filter tag_list
target_names = [tag_list[idx] for idx in unique_labels]
print(classification_report(y_test, y_pred, target_names=target_names, labels=unique_labels))