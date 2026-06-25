import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import joblib
import os  


os.makedirs('models', exist_ok=True)
os.makedirs('assets', exist_ok=True)

import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🔬 BREAST CANCER CLASSIFIER - TRAINING")
print("="*70)

# 1. LOAD DATASET 
print("\n📊 Loading Breast Cancer Dataset...")
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
feature_names = cancer.feature_names
target_names = cancer.target_names

print(f"✅ Dataset loaded!")
print(f"   • Total samples: {X.shape[0]}")
print(f"   • Features: {X.shape[1]}")
print(f"   • Classes: {target_names[0]} (0) and {target_names[1]} (1)")
print(f"   • Benign: {sum(y==0)} samples")
print(f"   • Malignant: {sum(y==1)} samples")

#  2. SPLIT DATA 
print("\n🔄 Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   • Training: {X_train.shape[0]} samples")
print(f"   • Testing: {X_test.shape[0]} samples")

#  3. SCALE FEATURES 
print("\n📏 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled!")

# 4. TRAIN MODELS 
print("\n🤖 Training models...")

models = {
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

param_grids = {
    'KNN': {'n_neighbors': [3, 5, 7, 9]},
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']},
    'Decision Tree': {'max_depth': [3, 5, 7, 10]},
    'Random Forest': {'n_estimators': [50, 100], 'max_depth': [5, 10]},
    'Logistic Regression': {'C': [0.1, 1, 10]}
}

results = {}
best_models = {}

for name, model in models.items():
    print(f"\n   🔍 Training {name}...")
    
    grid = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    
    best_model = grid.best_estimator_
    best_models[name] = best_model
    
    y_pred = best_model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {
        'Best Params': str(grid.best_params_),
        'Test Accuracy': f"{test_acc:.2%}",
        'F1 Score': f"{f1:.2%}"
    }
    
    print(f"  Test accuracy: {test_acc:.2%}")

# 5. FIND BEST MODEL 
print("\n" + "="*70)
print("🏆 RESULTS")
print("="*70)

results_df = pd.DataFrame(results).T
print(results_df)

best_model_name = max(results, key=lambda x: float(results[x]['Test Accuracy'].rstrip('%')))
print(f"\n🌟 BEST MODEL: {best_model_name}")
print(f"   Accuracy: {results[best_model_name]['Test Accuracy']}")

# 6. SAVE MODEL 
print("\n💾 Saving model...")
best_model = best_models[best_model_name]


joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("Model saved successfully!")

# 7. EVALUATE BEST MODEL 
print("\n📊 Evaluating best model...")
y_pred = best_model.predict(X_test_scaled)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"   TP: {cm[1,1]}  FP: {cm[0,1]}")
print(f"   FN: {cm[1,0]}  TN: {cm[0,0]}")

# Classification Report
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('assets/confusion_matrix.png', dpi=150)
print("📈 Confusion matrix saved!")

# Feature Importance (if available)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[-10:]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(10), importances[indices], color='teal')
    plt.yticks(range(10), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title(f'Top 10 Features - {best_model_name}')
    plt.tight_layout()
    plt.savefig('assets/feature_importance.png', dpi=150)
    print("📈 Feature importance saved!")

print("\n" + "="*70)
print(" TRAINING COMPLETE!")
print(f" Best Model: {best_model_name}")
print("="*70)
