from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

print(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# ----- Manual Input -----
print("\nEnter feature values for prediction (sepal length, sepal width, petal length, petal width):")
try:
    sl = float(input("Sepal length (cm): "))
    sw = float(input("Sepal width (cm): "))
    pl = float(input("Petal length (cm): "))
    pw = float(input("Petal width (cm): "))

    manual_input = [[sl, sw, pl, pw]]
    prediction = model.predict(manual_input)[0]
    class_name = iris.target_names[prediction]

    print(f"\nPredicted Iris Class: {class_name.title()}")
except ValueError:
    print("Invalid input. Please enter numeric values.")
