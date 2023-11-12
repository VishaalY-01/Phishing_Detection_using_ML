import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import tensorflow as tf
from keras.models import load_model

# def predict(phishy):
# Load and preprocess your dataset
data = pd.read_csv('phishing_dataset.csv')

# Rename columns to match the dataset structure
data = data.rename(columns={'label': 'label', 'text': 'text'})

data['text'] = data['text'].apply(lambda x: x.lower())
data['label'] = data['label'].apply(lambda x: 0 if x == 'ham' else 1)

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Naive Bayes
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vectorized, y_train)

# SVM
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_vectorized, y_train)

# Deep Learning
def create_deep_learning_model():
    model = Sequential()
    model.add(Dense(128, input_dim=X_train_vectorized.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

deep_learning_model = create_deep_learning_model()

# Manually split your data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_vectorized, y_train, test_size=0.2, random_state=42)

# Convert Scipy CSR matrices to NumPy arrays for Keras
X_train_split_array = X_train_split.toarray()
X_val_split_array = X_val_split.toarray()

deep_learning_model.fit(X_train_split_array, y_train_split, epochs=10, batch_size=32, validation_data=(X_val_split_array, y_val_split))

deep_learning_model.save('phishing_model.keras')

# Load the saved model
loaded_model = load_model('phishing_model.keras')

# Email input by user
user_email = input("Enter the content")
user_email_vectorized = vectorizer.transform([user_email])

# Predict using the loaded model
deep_learning_prediction = loaded_model.predict(user_email_vectorized)

# Convert prediction to 0 or 1
if deep_learning_prediction[0][0] > 0.5:
    deep_learning_prediction = 1
else:
    deep_learning_prediction = 0

if deep_learning_prediction == 1:
    res="This given email is a phishing email."
else:
    res="This given email is not a phishing email."

print(res)
# return res
