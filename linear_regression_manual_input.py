import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data
iris = load_iris()
X = iris.data[:, 1:]  # Use all features except 'sepal length' for prediction
y = iris.data[:, 0]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_:.2f}")

# Visualization
plt.scatter(y_test, y_pred, color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel("Actual Sepal Length")
plt.ylabel("Predicted Sepal Length")
plt.title("Linear Regression on Iris Dataset")
plt.show()

# Manual input for prediction
print("\nEnter feature values for prediction (sepal width, petal length, petal width):")
try:
    sw = float(input("Sepal width (cm): "))
    pl = float(input("Petal length (cm): "))
    pw = float(input("Petal width (cm): "))

    manual_input = [[sw, pl, pw]]
    prediction = model.predict(manual_input)[0]

    print(f"\nPredicted Sepal Length: {prediction:.2f} cm")

except ValueError:
    print("Invalid input. Please enter numeric values.")
