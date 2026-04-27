import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("dataset.csv")
df['label'] = df['label'].map({'phishing': 1, 'safe': 0})
df = df.dropna()

# Features
X = df['text']
y = df['label']

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show(block=False)

# Test loop
while True:
    msg = input("\nEnter email (type 'exit' to quit): ")

    if msg.lower() == "exit":
        print("Exiting...")
        break

    vec = vectorizer.transform([msg])
    result = model.predict(vec)

    if result[0] == 1:
        print("⚠️ Phishing Email")
    else:
        print("✅ Safe Email")
