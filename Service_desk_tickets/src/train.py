import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)

# --- Configuration ---
MODEL_NAME = "distilbert-base-uncased"
DATA_DIR = "data"
OUTPUT_DIR = "models/distilbert-ticket-classifier"
NUM_LABELS = 3
LABEL_MAP = {'Hardware': 0, 'Billing': 1, 'Access': 2}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {'accuracy': acc, 'f1': f1}

class TicketDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_model():
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    # Map labels
    train_df['label_id'] = train_df['label'].map(LABEL_MAP)
    test_df['label_id'] = test_df['label'].map(LABEL_MAP)
    
    # Preprocessing
    print(f"Tokenizing data using {MODEL_NAME}...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    
    train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True, max_length=64)
    test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=64)
    
    train_dataset = TicketDataset(train_encodings, train_df['label_id'].tolist())
    val_dataset = TicketDataset(test_encodings, test_df['label_id'].tolist())
    
    # Model Setup
    print("Initializing model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=NUM_LABELS,
        id2label={v: k for k, v in LABEL_MAP.items()},
        label2id=LABEL_MAP
    )
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,              # 3 epochs
        per_device_train_batch_size=8,  # Small batch size for CPU/Compat
        per_device_eval_batch_size=16,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        no_cuda=False if torch.cuda.is_available() else True # Use CPU if no CUDA
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Final Evaluation
    eval_result = trainer.evaluate()
    print(f"Final Evaluation Results: {eval_result}")
    
    # Save metrics for MLOps logic gate
    with open(os.path.join(OUTPUT_DIR, "eval_metrics.txt"), "w") as f:
        f.write(str(eval_result))

if __name__ == "__main__":
    train_model()
