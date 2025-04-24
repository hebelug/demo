import json
import random
import logging
import os
from typing import List, Dict, Tuple
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"dualcl_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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
class DualCLRelationDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer: BertTokenizer, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = RELATION_LABELS
        # Validate data
        for idx, item in enumerate(self.data):
            if not item.get("text") or not item.get("label"):
                logger.warning(f"Invalid data at index {idx}: text={item.get('text')}, label={item.get('label')}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = item["label"]
        label_id = LABEL2ID[label]

        # Randomly shuffle labels to mitigate position bias
        labels = self.labels.copy()
        random.shuffle(labels)
        label_tokens = " ".join(labels)
        true_label_pos = labels.index(label)  # Position of true label in shuffled list

        # Create input sequence: [CLS] label1 label2 ... [SEP] text [SEP]
        input_text = f"{label_tokens} [SEP] {text}"
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Find token indices for labels
        label_indices = []
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze())
        current_pos = 0
        for l in labels:
            l_tokens = self.tokenizer.tokenize(l)
            start = current_pos + 1  # Skip [CLS]
            current_pos = start + len(l_tokens)
            label_indices.append(start)

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label_id": torch.tensor(label_id, dtype=torch.long),
            "true_label_pos": torch.tensor(true_label_pos, dtype=torch.long),
            "label_indices": torch.tensor(label_indices, dtype=torch.long)
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

# DualCL Model
class DualCLModel(torch.nn.Module):
    def __init__(self, num_labels):
        super(DualCLModel, self).__init__()
        self.bert = BertModel.from_pretrained("/home/f/bert/biobert-v1.2")
        self.num_labels = num_labels
        self.hidden_size = self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask, label_indices):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # First element of tuple is last_hidden_state [batch_size, seq_len, hidden_size]

        # Extract z_i from [CLS] token (position 0)
        z_i = sequence_output[:, 0, :]  # [batch_size, hidden_size]
        # Normalize z_i to unit vector
        z_i = F.normalize(z_i, dim=-1)

        # Extract θ_i^k for each label from label token positions
        theta_i = torch.zeros(input_ids.size(0), self.num_labels, self.hidden_size).to(device)
        for i in range(self.num_labels):
            theta_i[:, i, :] = sequence_output[torch.arange(input_ids.size(0)), label_indices[:, i], :]
        # Normalize theta_i to unit vectors
        theta_i = F.normalize(theta_i, dim=-1)

        return z_i, theta_i

# Vectorized contrastive loss functions
def contrastive_loss_z(z_i, theta_i, true_label_pos, tau=0.1, eps=1e-8):
    batch_size = z_i.size(0)
    # Compute similarities for true labels
    true_label_indices = true_label_pos.unsqueeze(-1)  # [batch_size, 1]
    theta_star = theta_i[torch.arange(batch_size), true_label_pos]  # [batch_size, hidden_size]
    
    # Compute positive similarities
    pos_sim = torch.exp(torch.sum(z_i * theta_star, dim=-1) / tau)  # [batch_size]
    
    # Compute all similarities (including negative)
    all_sim = torch.exp(torch.bmm(z_i.unsqueeze(1), theta_i.transpose(1, 2)).squeeze(1) / tau)  # [batch_size, num_labels]
    all_sim_sum = all_sim.sum(dim=-1) + eps  # [batch_size]
    
    # Compute loss
    loss = -torch.log(pos_sim / all_sim_sum + eps)
    return loss.mean()

def contrastive_loss_theta(z_i, theta_i, true_label_pos, tau=0.1, eps=1e-8):
    batch_size = z_i.size(0)
    # Compute similarities for true labels
    theta_star = theta_i[torch.arange(batch_size), true_label_pos]  # [batch_size, hidden_size]
    
    # Compute positive similarities
    pos_sim = torch.exp(torch.sum(theta_star * z_i, dim=-1) / tau)  # [batch_size]
    
    # Compute all similarities (including negative)
    all_sim = torch.exp(torch.bmm(theta_star.unsqueeze(1), z_i.unsqueeze(-1)).squeeze(-1) / tau)  # [batch_size, batch_size]
    all_sim_sum = all_sim.sum(dim=-1) + eps  # [batch_size]
    
    # Compute loss
    loss = -torch.log(pos_sim / all_sim_sum + eps)
    return loss.mean()

# Training function
def train_model(model, train_loader, val_loader, optimizer, epochs: int = 3, lambda_dual: float = 0.1, model_save_path: str = "models"):
    best_f1 = 0.0
    os.makedirs(model_save_path, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_ce_loss = 0
        total_dual_loss = 0

        # Add tqdm progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}", leave=False)
        for batch in train_pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_ids = batch["label_id"].to(device)
            true_label_pos = batch["true_label_pos"].to(device)
            label_indices = batch["label_indices"].to(device)

            optimizer.zero_grad()
            z_i, theta_i = model(input_ids, attention_mask, label_indices)

            # Cross-entropy loss
            logits = torch.bmm(theta_i, z_i.unsqueeze(-1)).squeeze(-1)  # [batch_size, num_labels]
            ce_loss = F.cross_entropy(logits, label_ids)

            # Dual contrastive loss
            loss_z = contrastive_loss_z(z_i, theta_i, true_label_pos)
            loss_theta = contrastive_loss_theta(z_i, theta_i, true_label_pos)
            dual_loss = loss_z + loss_theta

            # Overall loss
            loss = ce_loss + lambda_dual * dual_loss

            # Check for NaN loss
            if torch.isnan(loss):
                logger.warning("NaN loss detected, skipping batch")
                continue

            loss.backward()
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_dual_loss += dual_loss.item()

            # Update tqdm progress bar with current loss
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        avg_ce_loss = total_ce_loss / len(train_loader)
        avg_dual_loss = total_dual_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{epochs} - Total Loss: {avg_loss:.4f}, CE Loss: {avg_ce_loss:.4f}, Dual Loss: {avg_dual_loss:.4f}")

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

    # Add tqdm progress bar for evaluation
    eval_pbar = tqdm(data_loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in eval_pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            label_ids = batch["label_id"].to(device)
            label_indices = batch["label_indices"].to(device)

            z_i, theta_i = model(input_ids, attention_mask, label_indices)
            logits = torch.bmm(theta_i, z_i.unsqueeze(-1)).squeeze(-1)  # [batch_size, num_labels]
            preds = torch.argmax(logits, dim=1)

            true_labels.extend(label_ids.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    f1 = f1_score(true_labels, pred_labels, average="macro")
    precision = precision_score(true_labels, pred_labels, average="macro")
    recall = recall_score(true_labels, pred_labels, average="macro")

    return f1, precision, recall

# Main function
def main():
    # Load data
    data = load_data("DDI_Test.json")
    logger.info(f"Loaded {len(data)} data samples")

    # Split data
    train_data, val_data, test_data = split_data(data)
    logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("/home/f/bert/biobert-v1.2")
    model = DualCLModel(num_labels=len(RELATION_LABELS)).to(device)

    # Create datasets and dataloaders
    train_dataset = DualCLRelationDataset(train_data, tokenizer)
    val_dataset = DualCLRelationDataset(val_data, tokenizer)
    test_dataset = DualCLRelationDataset(test_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

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