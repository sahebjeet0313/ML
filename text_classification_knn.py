from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Sample training dataset
texts = [
    "I love this movie, it was fantastic!",
    "Amazing acting and great story.",
    "What a wonderful experience.",
    "This film was terrible and boring.",
    "Worst movie I’ve ever seen.",
    "The plot was dull and disappointing.",
    "An excellent, must-watch film!",
    "I hated the acting and direction.",
    "Brilliant performance by the cast.",
    "Awful! Not worth the time."
]

labels = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]  # 1: Positive, 0: Negative

# Vectorize using CountVectorizer (no TF-IDF)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Split into train/test (to simulate training)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# ----- Manual Input -----
manual_text = input("Enter a sentence to classify sentiment: ")

# Vectorize the input using same vocabulary
manual_vector = vectorizer.transform([manual_text])

# Predict sentiment
prediction = knn.predict(manual_vector)[0]
label = "Positive 😊" if prediction == 1 else "Negative 😠"

print(f"\nPredicted Sentiment: {label}")
