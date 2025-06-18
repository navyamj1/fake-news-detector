import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("news.csv")

df = df.dropna(subset=['text', 'label'])
X = df['text'].astype(str)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tfidf = TfidfVectorizer(stop_words='english', max_df=0.7, ngram_range=(1, 2))
tfidf_train = tfidf.fit_transform(X_train)
tfidf_test = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(tfidf_train, y_train)

y_pred = model.predict(tfidf_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
