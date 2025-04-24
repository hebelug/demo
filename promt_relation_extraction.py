import json
import random
import logging
import os
from typing import List, Dict, Tuple
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np
from uuid import uuid4

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define relation labels
RELATION_LABELS = ["NA", "effect", "mechanism", "advise", "int"]
LABEL2ID = {label: idx for idx, label in enumerate(RELATION_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

# Dataset class
class RelationExtractionDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: BertTokenizer, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = item["label"]

        # Create prompt
        # Assume two entities are present; for simplicity, use placeholders
        # In practice, entity extraction would be needed
        prompt = f"Does the interaction between drugs in the sentence '{text}' result in {label}?"
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(LABEL2ID[label], dtype=torch.long)
        }

# Load and preprocess data
def load_data(file_path: str) -> List[Dict]:
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def split_data(data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1):
    train_data, temp_data = train_test_split(data, train_size=train_ratio, random_state=42)
    val_data, test_data = train_test_split(temp_data, train_size=val_ratio/(1-train_ratio), random_state=42)
    return train_data, val_data, test_data

# Training function
def train_model(model, train_loader, val_loader, optimizer, epochs: int = 3, model_save_path: str = "models"):
    best_f1 = 0.0
    os.makedirs(model_save_path, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_loss:.4f}")

        # Validation
        f1, precision, recall = evaluate_model(model, val_loader)
        logger.info(f"Epoch {epoch+1}/{epochs} - Validation F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        # Save model if F1 improves
        if f1 > best_f1:
            best_f1 = f1
            model_save_file = os.path.join(model_save_path, f"best_model_f1_{f1:.4f}.pt")
            torch.save(model.state_dict(), model_save_file)
            logger.info(f"Saved model with F1: {f1:.4f} at {model_save_file}")

    return best_f1

# Evaluation function
def evaluate_model(model, data_loader) -> Tuple[float, float, float]:
    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    f1 = f1_score(true_labels, pred_labels, average="macro")
    precision = precision_score(true_labels, pred_labels, average="macro")
    recall = recall_score(true_labels, pred_labels, average="macro")

    return f1, precision, recall

# Main function
def main():
    # Load data
    # Assuming the JSON is saved as 'DDI_Test.json'
    data = load_data("DDI_Test.json")
    logger.info(f"Loaded {len(data)} data samples")

    # Split data
    train_data, val_data, test_data = split_data(data)
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(RELATION_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    ).to(device)

    # Create datasets and dataloaders
    train_dataset = RelationExtractionDataset(train_data, tokenizer)
    val_dataset = RelationExtractionDataset(val_data, tokenizer)
    test_dataset = RelationExtractionDataset(test_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Train model
    logger.info("Starting training...")
    best_f1 = train_model(model, train_loader, val_loader, optimizer)

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_f1, test_precision, test_recall = evaluate_model(model, test_loader)
    logger.info(f"Test F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

    # Save final model
    final_model_path = os.path.join("models", f"final_model_f1_{test_f1:.4f}.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model at {final_model_path}")

if __name__ == "__main__":
    main()