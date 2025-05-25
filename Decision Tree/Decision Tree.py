import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn import tree

# Load the dataset
data = pd.read_csv("students.csv")  

# Drop student_id i do not need it
data = data.drop('student_id', axis=1)

# Convert categorical column to numeric (yes → 1, no → 0)
data['school_support'] = data['school_support'].map({'yes': 1, 'no': 0})

# Separate features and target
X = data[['study_hours', 'absences', 'school_support']]
y = data['final_grade']

# Build decision tree model
model = DecisionTreeClassifier()
model.fit(X, y)

# Predict on training data
y_pred = model.predict(X)

# Evaluate
print("R² Score:", r2_score(y, y_pred))

# Optional: Visualize the decision tree
plt.figure(figsize=(12, 6))
tree.plot_tree(model, feature_names=X.columns, filled=True, rounded=True)
plt.title("Decision Tree for Final Grade Prediction")
plt.show()
