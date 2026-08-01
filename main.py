import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
import pickle
# Load the dataset
data = pd.read_csv("dataset.csv")

# Display the first 5 rows
print("First 5 Rows:")
print(data.head())

# Display information about the dataset
print("\nDataset Information:")
data.info()

# Display the dataset size
print("\nDataset Shape:")
print(data.shape)
# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())
# Separate input and output
X = data["review"]
y = data["label"]

# Convert text into numbers using TF-IDF
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

print("\nTF-IDF Conversion Successful!")
print("Shape of X:", X.shape)
# Split dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nDataset Split Successful!")
print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)
# Train the Naive Bayes model

model = MultinomialNB()

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")
# Predict the test data

y_pred = model.predict(X_test)

print("\nPrediction Completed!")

print("Predicted Labels:")
print(y_pred)

print("\nActual Labels:")
print(y_test.values)
# Calculate accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)
# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)
# Classification Report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Save the trained model

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully as model.pkl")
# Take review from user
review = input("\nEnter a product review: ")

# Convert the review to TF-IDF
review_vector = vectorizer.transform([review])

# Predict
prediction = model.predict(review_vector)

# Display result
if prediction[0] == 1:
    print("Prediction: Fake Review")
else:
    print("Prediction: Genuine Review")