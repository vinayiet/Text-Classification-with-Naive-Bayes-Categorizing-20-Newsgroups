from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def load_data():
    """
    Load the 20 Newsgroups dataset and split it into training and testing sets.

    Returns:
        X_train (list): Training data.
        X_test (list): Testing data.
        y_train (list): Training labels.
        y_test (list): Testing labels.
    """
    # Load the 20 Newsgroups dataset
    data = fetch_20newsgroups(subset='all', categories=None, shuffle=True, random_state=42)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25, random_state=33)

    return X_train, X_test, y_train, y_test

# Load the data
X_train, X_test, y_train, y_test = load_data()

# Initialize a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Transform the training data into TF-IDF features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Initialize a Multinomial Naive Bayes classifier
clf = MultinomialNB()

# Train the classifier on the TF-IDF features
clf.fit(X_train_tfidf, y_train)

# Transform the testing data into TF-IDF features
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Predict the labels for the testing data
y_pred = clf.predict(X_test_tfidf)

# Print classification report
print(classification_report(y_test, y_pred))
