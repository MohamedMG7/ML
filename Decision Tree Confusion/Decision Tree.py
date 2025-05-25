import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

# Data Cleaning
# Load the dataset
data = pd.read_csv("students.csv")

# Drop student_id
data = data.drop('student_id', axis=1)

# Convert school_support to numeric
data['school_support'] = data['school_support'].map({'yes': 1, 'no': 0})

# Convert final_grade to pass/fail classification
data['grade_class'] = data['final_grade'].apply(lambda x: 'pass' if x >= 50 else 'fail')

# Features and target
X = data[['study_hours', 'absences', 'school_support']]
y = data['grade_class']

# Encode class labels (pass = 1, fail = 0)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train classifier
model = DecisionTreeClassifier()
model.fit(X, y_encoded)

# Predict
y_pred = model.predict(X)

# Evaluate
print(classification_report(y_encoded, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_encoded, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()

# Visualize Decision Tree
plt.figure(figsize=(12, 6))
tree.plot_tree(model, feature_names=X.columns, class_names=le.classes_, filled=True, rounded=True)
plt.title("Decision Tree: Pass/Fail Prediction")
plt.show()
