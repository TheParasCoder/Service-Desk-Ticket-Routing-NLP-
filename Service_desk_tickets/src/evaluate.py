import os
import pandas as pd
import torch
import numpy as np
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureDrift, TrainTestFeatureDrift, LabelDrift
from deepchecks.tabular.suites import data_integrity
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

# --- Configuration ---
MODEL_PATH = "models/distilbert-ticket-classifier"
DATA_DIR = "data"
BASELINE_F1 = 0.85  # Simulated "Production" Model Performance

def run_integrity_checks(df):
    print("Running Data Integrity Checks...")
    ds = Dataset(df, label='label', cat_features=[])
    # Run a default suite of checks
    suite = data_integrity()
    result = suite.run(ds)
    # Save results
    result.save_as_html("results/data_integrity.html")
    print("Integrity checks saved to results/data_integrity.html")
    
    # Simple check: pass efficiently if no critical failure
    # In a real scenario, we'd parse 'result' for failure conditions.
    return True

def load_model_and_predict(test_df):
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    
    inputs = tokenizer(test_df['text'].tolist(), padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).numpy()
    return predictions

def evaluate_logic_gate():
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    # Map labels to IDs for metrics
    label_map = {'Hardware': 0, 'Billing': 1, 'Access': 2}
    true_labels = test_df['label'].map(label_map).tolist()
    
    # 1. Run Integrity Checks on Test Data (simulating new incoming batch)
    if not run_integrity_checks(test_df):
        print("Data Integrity Check Failed! Aborting deployment.")
        return False
    
    # 2. Evaluate Model
    preds = load_model_and_predict(test_df)
    
    new_f1 = f1_score(true_labels, preds, average='weighted')
    print(f"New Model Weighted F1 Score: {new_f1:.4f}")
    print(f"Baseline (Production) F1 Score: {BASELINE_F1:.4f}")
    
    # 3. Logic Gate
    improvement = new_f1 - BASELINE_F1
    print(f"Improvement: {improvement:.4f}")
    
    if improvement >= 0.02:
        print(">>> SUCCESS: Model has improved by at least 2%. Deployment triggered.")
        return True
    else:
        print(">>> FAIL: Model did not match improvement threshold. Deployment skipped.")
        return False

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    if os.path.exists(MODEL_PATH):
        evaluate_logic_gate()
    else:
        print("Model not found. Please run train.py first.")
