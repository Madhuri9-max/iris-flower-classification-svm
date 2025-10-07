
# Importing required libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

iris = datasets.load_iris()
X = iris.data
y = iris.target

# Convert to DataFrame for easy understanding (optional)
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in y]

print("First five rows of the dataset:")
print(df.head())


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# You can change the kernel to 'linear', 'rbf', or 'poly'
model = SVC(kernel='linear', C=1.0, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion Matrix Visualization
plt.figure(figsize=(6, 4))
sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    cmap='Greens',
    fmt='d',
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)
plt.title('Confusion Matrix - Iris Flower Classification (SVM)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

sample = [[5.1, 3.5, 1.4, 0.2]]  # Example: Setosa
sample_scaled = scaler.transform(sample)
predicted_class = iris.target_names[model.predict(sample_scaled)[0]]
print(f"\nPredicted Class for sample {sample}: {predicted_class}")


