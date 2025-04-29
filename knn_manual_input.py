from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
print(target_names)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict on test data
y_pred = knn.predict(X_test)

# Accuracy
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}\n")

# Print predictions and check correctness
print("Index\tActual\t\tPredicted\tCorrect?")
for i in range(len(y_test)):
    actual = target_names[y_test[i]]
    predicted = target_names[y_pred[i]]
    is_correct = "✅" if y_test[i] == y_pred[i] else "❌"
    print(f"{i}\t{actual:<15}{predicted:<15}{is_correct}")

# Manual input for prediction
print("Enter the features for prediction:")
sepal_length = float(input("Sepal length (cm): "))
sepal_width = float(input("Sepal width (cm): "))
petal_length = float(input("Petal length (cm): "))
petal_width = float(input("Petal width (cm): "))

manual_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
manual_pred = knn.predict(manual_input)
print(f"Predicted class: {target_names[manual_pred][0]}")

# Plotting the dataset and manual input
plt.figure(figsize=(8, 6))

# Plot all data points with color by class
for i, target_name in enumerate(target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], label=target_name)

# Plot manual input point
plt.scatter(manual_input[0, 0], manual_input[0, 1], color='red', marker='X', s=100, label='Manual Input')

plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('Iris Dataset - Sepal Length vs Sepal Width')
plt.legend()
plt.show()
