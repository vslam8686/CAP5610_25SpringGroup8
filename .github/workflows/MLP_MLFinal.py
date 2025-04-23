import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dataset_grouped_selected_balanced.csv")
drop_cols = ['number_on_set', 'track_id', 'artists', 'album_name', 'track_name', 'track_genre']
df = df.drop(columns=drop_cols)

# Features
df['duration_minutes'] = df['duration_ms'] / 60000
df['loudness_scaled'] = df['loudness'] + 60
df['energy_dance_ratio'] = df['energy'] / (df['danceability'] + 1e-5)

# Onehott
df['explicit'] = df['explicit'].astype(int)
categorical = ['key', 'mode', 'time_signature']
df = pd.get_dummies(df, columns=categorical, drop_first=True)

# Labels for the
X = df.drop(columns=['track_genre_grouped'])
y = df['track_genre_grouped']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train & test splitting
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

param_grid = {
    'hidden_layer_sizes': [(128,), (128, 64), (256, 128)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.001],
    'learning_rate': ['adaptive'],
    'max_iter': [500],
    'early_stopping': [True]
}

mlp = MLPClassifier(random_state=42)
grid = GridSearchCV(mlp, param_grid, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)


best_mlp = grid.best_estimator_
y_pred = best_mlp.predict(X_test)

print("\nBest Parameters:", grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("lassification & debug:\n", classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred, labels=best_mlp.classes_)
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=best_mlp.classes_, yticklabels=best_mlp.classes_, cmap='Blues')
plt.title("Confusion Matrix with Genre Labels")
plt.xlabel("Predicted Genre")
plt.ylabel("True Genre")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# bar plot
report_dict = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()
df_report = df_report.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
df_report[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(12, 6))
plt.title("Precision, Recall, and F1-Score per Genre")
plt.ylabel("Score")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Training Loss vs Epoch
if hasattr(best_mlp, 'loss_curve_'):
    plt.figure(figsize=(8, 5))
    plt.plot(best_mlp.loss_curve_, label='Training Loss')
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

avg_loss = np.mean(best_mlp.loss_curve_)
print("Average Training Loss:", avg_loss)
final_loss = best_mlp.loss_curve_[-1]
print("Final Loss:", final_loss)

# Validation Accuracy per Epoch
if hasattr(best_mlp, 'validation_scores_'):
    plt.figure(figsize=(8, 5))
    plt.plot(best_mlp.validation_scores_, label='Validation Accuracy')
    plt.title("Validation Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()