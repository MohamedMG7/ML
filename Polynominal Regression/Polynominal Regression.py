import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# Load your data
data = pd.read_csv("ice_cream.csv")  # Or use pd.read_clipboard() if pasted

X = data[['Temperature (°C)']]
y = data['Ice Cream Sales (units)']

# Try a polynomial of degree 2 (quadratic)
degree = 2
model = make_pipeline(PolynomialFeatures(degree), LinearRegression()) # automatically pass to linear regression
model.fit(X, y)

# Predict values over a smooth temperature range
y_pred = model.predict(X)

# Plot
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label=f'Polynomial Degree {degree}')
plt.xlabel("Temperature (°C)")
plt.ylabel("Ice Cream Sales (units)")
plt.title("Polynomial Regression: Temperature vs Ice Cream Sales")
plt.legend()
plt.grid(True)
plt.show()

#evaluation
r2 = r2_score(y, y_pred)
print(f"R² Score: {r2:.4f}")
print("Polynomial coefficients:", model.named_steps['linearregression'].coef_)
print("Intercept:", model.named_steps['linearregression'].intercept_)