##1.Need to install pandas, pytorch, sklearn, matplotlib, numpy.

import pandas as pd
import os
from pytorch_tabular import TabularModel
from pytorch_tabular.models.ft_transformer.config import FTTransformerConfig
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_recall_fscore_support
)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from pytorch_lightning.callbacks import Callback

##2.Configurations of Paths
DATA_PATH = "dataset_grouped_selected_balanced.csv"
TARGET_COLUMN = "track_genre_encoded"
ORIGINAL_GENRE_COLUMN = "track_genre_grouped"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

##3.Loading of paths defined on the configuration(2)
df = pd.read_csv(DATA_PATH)
le = LabelEncoder()
df[TARGET_COLUMN] = le.fit_transform(df[ORIGINAL_GENRE_COLUMN])
genre_names = le.classes_
##4. Define the split 81/10/9
train_val, test = train_test_split(df, test_size=0.1, stratify=df[TARGET_COLUMN], random_state=42)
train, val = train_test_split(train_val, test_size=0.1, stratify=train_val[TARGET_COLUMN], random_state=42)
##5. Define and separate categorical categories between continuous and categorical
categorical_cols = ["explicit", "key", "mode", "time_signature"]
continuous_cols = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

##6. Configurate the columns to be selected as categorical and normalize the continuous
data_config = DataConfig(
    target=[TARGET_COLUMN],
    continuous_cols=continuous_cols,
    categorical_cols=categorical_cols,
    normalize_continuous_features=True,
)
##7. Define the hyperparameters for the model.
model_config = FTTransformerConfig(
    task="classification",
    learning_rate=5e-4,
    metrics=["accuracy"],
    num_attn_blocks=4,
    attn_dropout=0.1,
    ff_dropout=0.1,
    input_embed_dim=64,
    num_heads=4
)
##8. Define the epochs and enable the gpu for faster training.
trainer_config = TrainerConfig(
    max_epochs=100, ##100 and 80 epochs were the best results, but I kept 100.
    accelerator="gpu", #used a rtx 3060
    devices=1,
    auto_lr_find=False, ##for previous runs I tried using this
    checkpoints="valid_loss",
    early_stopping=None,  ##disable early stopping, used for checking overfitting
    progress_bar="console",
)

optimizer_config = OptimizerConfig()

##9. Configurate the callback for login the data.
class AccuracyLogger(Callback):
    def __init__(self):
        self.val_accuracies = []

    def on_validation_epoch_end(self, trainer, pl_module):
        val_acc = trainer.callback_metrics.get("valid_accuracy")
        if val_acc is not None:
            self.val_accuracies.append(val_acc.item())

acc_logger = AccuracyLogger()

##10. Initialize the model
print("Initializing model")
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    trainer_config=trainer_config,
    optimizer_config=optimizer_config,
)

##11. Train the model
print("Training model")
tabular_model.fit(train=train, validation=val, callbacks=[acc_logger])

##12. Evaluate the model
print("Evaluating")
result = tabular_model.evaluate(test=test)
print("Evaluation:", result)

##13.Print predictions
pred_df = tabular_model.predict(test)
pred_df.to_csv(os.path.join(RESULTS_DIR, "predictions_selected.csv"), index=False)

##14.Generate confussion matrix
y_true = test[TARGET_COLUMN]
y_pred = pred_df[f"{TARGET_COLUMN}_prediction"]
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(12, 12))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=genre_names)
disp.plot(ax=ax, xticks_rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_selected.png"))
plt.close()

##15. Print Accuracy by genre
genre_acc = (
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    .groupby("y_true")
    .apply(lambda x: accuracy_score(x["y_true"], x["y_pred"]))
    .reset_index()
    .rename(columns={0: "Accuracy", "y_true": "Genre_Code"})
    .sort_values("Accuracy", ascending=False)
)
genre_acc["Genre"] = genre_names[genre_acc["Genre_Code"]]
genre_acc.to_csv(os.path.join(RESULTS_DIR, "accuracy_by_genre_selected.csv"), index=False)
print("Accuracy by genre saved.")

##16. Print the barplot of accuracy per genre
plt.figure(figsize=(12, 6))
plt.barh(genre_acc["Genre"], genre_acc["Accuracy"])
plt.xlabel("Accuracy")
plt.title("Accuracy by Genre")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "accuracy_by_genre_barplot.png"))
plt.close()

##17. Print the Precission/Recall/F1 scores
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average=None, zero_division=0
)

metrics_df = pd.DataFrame({
    "Genre": genre_names,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1
})
metrics_df = metrics_df.sort_values("F1-Score", ascending=False)
metrics_df.to_csv(os.path.join(RESULTS_DIR, "precision_recall_f1_by_genre.csv"), index=False)
print("Precision/Recall/F1 saved.")

##18.(Optional)Print top and bottom 5 scores by F1
print("Top 5 Genres by F1-Score:")
print(metrics_df.head(5).to_string(index=False))

print("Bottom 5 Genres by F1-Score:")
print(metrics_df.tail(5).to_string(index=False))

##19. Print wighted scores.
macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
weighted = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
print(f"Macro F1: {macro[2]:.3f}, Weighted F1: {weighted[2]:.3f}")

##20. Print accuracy val curve.
if acc_logger.val_accuracies:
    plt.figure()
    plt.plot(acc_logger.val_accuracies)
    plt.title("Validation Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.savefig(os.path.join(RESULTS_DIR, "val_accuracy_curve.png"))
    plt.close()
    print("Validation accuracy curve saved.")
