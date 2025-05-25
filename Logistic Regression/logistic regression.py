import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("HR.csv")

# Encode categorical columns
df['Department'] = LabelEncoder().fit_transform(df['Department'])
df['salary'] = LabelEncoder().fit_transform(df['salary'])

# Features and target
X = df.drop('left', axis=1)
y = df['left']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression model
model = LogisticRegression()  
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stayed', 'Left'])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression (Employee Left Prediction)")
plt.grid(False)
plt.show()
