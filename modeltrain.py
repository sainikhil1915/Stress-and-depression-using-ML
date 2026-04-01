import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

#############################################
# Part 1: Train on Depression Questionnaire
#############################################

print("📥 Loading depression dataset...")
depression_data = pd.read_csv('dataset/depressionDataset.csv')

# Drop rows with missing values
depression_data = depression_data.dropna()

# Add score column (sum of q1 to q10)
depression_data['score'] = depression_data[[f'q{i}' for i in range(1, 11)]].sum(axis=1)

# Features: q1 to q10 + score
X_depression = depression_data[[f'q{i}' for i in range(1, 11)] + ['score']]
y_depression = depression_data['class']

# Impute any missing values (just in case)
imputer = SimpleImputer(strategy='mean')
X_depression = imputer.fit_transform(X_depression)

# Split dataset
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_depression, y_depression, test_size=0.2, random_state=42)

# Define classifiers
depression_models = {
    'naive_bayes': GaussianNB(),
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'svm': SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced'),
    'mlp': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    'logistic_regression': LogisticRegression(max_iter=200, random_state=42, class_weight='balanced'),
    'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'decision_tree': DecisionTreeClassifier(random_state=42, class_weight='balanced')
}

print("🧠 Training models on depression dataset...")
for name, model in depression_models.items():
    model.fit(X_train_d, y_train_d)
    y_pred = model.predict(X_test_d)
    print(f"\n🔎 {name} - Depression Dataset Report")
    print(classification_report(y_test_d, y_pred))
    joblib.dump(model, f'models/{name}_depression.pkl')


#############################################
# Part 2: Train on Tweets Dataset
#############################################

print("\n📥 Loading tweets dataset...")
tweets = pd.read_csv('dataset/tweets.csv')
tweets = tweets[['message', 'label']].dropna()

X_tweets = tweets['message']
y_tweets = tweets['label']

# Vectorize tweets using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_tweets_vec = vectorizer.fit_transform(X_tweets)

# Save the vectorizer
joblib.dump(vectorizer, 'models/tfidf_vectorizer_tweets.pkl')

# Split dataset
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_tweets_vec, y_tweets, test_size=0.2, random_state=42)

# Define classifiers
tweet_models = {
    'naive_bayes': MultinomialNB(),
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'svm': SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced'),
    'mlp': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    'logistic_regression': LogisticRegression(max_iter=200, random_state=42, class_weight='balanced'),
    'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'decision_tree': DecisionTreeClassifier(random_state=42, class_weight='balanced')
}

print("🧠 Training models on tweets dataset...")
for name, model in tweet_models.items():
    model.fit(X_train_t, y_train_t)
    y_pred = model.predict(X_test_t)
    print(f"\n🔎 {name} - Tweets Dataset Report")
    print(classification_report(y_test_t, y_pred))
    joblib.dump(model, f'models/{name}_tweets.pkl')

print("\n✅ All models trained and saved successfully!")
