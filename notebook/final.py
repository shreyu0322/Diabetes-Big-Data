import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Set styles for plots
sns.set(style="whitegrid")

# Load the dataset
file_path = '../diabetes_012_health_indicators_BRFSS2021.csv'
diabetes_data = pd.read_csv(file_path)

# Display first few rows
diabetes_data.head()

# Data visualization
plt.figure(figsize=(8, 6))
sns.countplot(x='Diabetes_012', data=diabetes_data, palette='coolwarm')
plt.title("Diabetes Class Distribution")
plt.show()

# Check for missing values
diabetes_data.isnull().sum()

# Feature selection
X = diabetes_data.drop(columns=['Diabetes_012'])
y = diabetes_data['Diabetes_012']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train and evaluate models
def train_and_evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Save model
    with open(f'{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return accuracy

# Train models
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': GaussianNB()
}

accuracy_scores = {}
for name, model in models.items():
    accuracy_scores[name] = train_and_evaluate_model(model, name)

# ANN Model
ann_model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

ann_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
ann_model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, callbacks=[early_stopping], verbose=1)

# Evaluate ANN
y_pred_ann = np.argmax(ann_model.predict(X_test), axis=1)
accuracy_scores['ANN'] = accuracy_score(y_test, y_pred_ann)

# Save ANN model
ann_model.save('ANN_model.h5')

# Final accuracy scores
accuracy_scores
