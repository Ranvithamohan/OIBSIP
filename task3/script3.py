# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data = pd.read_csv('C:\Users\91779\Desktop\OIBSIP\OIBSIP\task3\archive3.csv')
X = data["v1"]
y = data['v2']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)
y_pred = classifier.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)
new_email = ["Congrats!!! You've won a million dollars! Click here to claim your prize."]
new_email_vectorized = vectorizer.transform(new_email)
prediction = classifier.predict(new_email_vectorized)
print("Prediction for the new email:", "Spam" if prediction[0] == 1 else "Not Spam")
