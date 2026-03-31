# ML Capstone Project:
# Naïve Bayes vs Logistic Regression Comparison
# Dataset: Iris Flower Dataset
# This script trains the two classification models and
# compares their performance using accuracy and
# confusion matrices.



# Step 1: Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Step 2: Load dataset (Iris flowers)
data = load_iris()
X = data.data      # features (flower measurements)
y = data.target    # labels (flower type)

# Step 3: Split data into training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 4: Train Naïve Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Step 5: Make predictions using Naïve Bayes
nb_predictions = nb_model.predict(X_test)

# Step 6: Train Logistic Regression model
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(X_train, y_train)

# Step 7: Make predictions using Logistic Regression
lr_predictions = lr_model.predict(X_test)

# Step 8: Calculate accuracy
nb_accuracy = accuracy_score(y_test, nb_predictions)
lr_accuracy = accuracy_score(y_test, lr_predictions)

print("Naïve Bayes Accuracy:", nb_accuracy)
print("Logistic Regression Accuracy:", lr_accuracy)

# Step 9:visualization (Line Plot)

models = ['Naïve Bayes', 'Logistic Regression']
accuracies = [nb_accuracy, lr_accuracy]

plt.plot(models, accuracies, marker='o')
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.9, 1.01)  # zoom in to see difference clearly
plt.grid()

# Show values on points
for i, value in enumerate(accuracies):
    plt.text(i, value, f"{value:.3f}", ha='center', va='bottom')

plt.show()

import pandas as pd  

# Create a table to compare model performance
results = pd.DataFrame({
    'Model': ['Naïve Bayes', 'Logistic Regression'],  # Names of models
    'Accuracy': [nb_accuracy, lr_accuracy]            # Their accuracy scores
})

print("\nComparison Table:")
print(results)  # Just to display the table in a clean format

from sklearn.metrics import confusion_matrix  
import matplotlib.pyplot as plt               
# Confusion matrices
nb_cm = confusion_matrix(y_test, nb_predictions)
lr_cm = confusion_matrix(y_test, lr_predictions)

# Plot Naïve Bayes heatmap
plt.figure()
plt.imshow(nb_cm)
plt.title("Naïve Bayes Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")

#  to add numbers inside boxes
for i in range(len(nb_cm)):
    for j in range(len(nb_cm)):
        plt.text(j, i, nb_cm[i, j], ha='center', va='center')

plt.show()

# Plot Logistic Regression heatmap
plt.figure()
plt.imshow(lr_cm)
plt.title("Logistic Regression Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")

# to add numbers inside boxes
for i in range(len(lr_cm)):
    for j in range(len(lr_cm)):
        plt.text(j, i, lr_cm[i, j], ha='center', va='center')

plt.show()




