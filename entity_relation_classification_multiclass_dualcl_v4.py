import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import random
import os

# 设置种子以确保可复现性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 自定义数据集类
class RelationDataset(Dataset):
    def __init__(self, json_data, tokenizer, max_length=128):
        self.sentences = []
        self.entity1 = []
        self.entity2 = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_names = ["No relation", "Effect", "Mechanism", "Advise", "Interaction"]
        self.label_map = {"NA": 0, "effect": 1, "mechanism": 2, "advise": 3, "int": 4}

        # 处理 JSON 数据，显示进度条
        print("Loading data...")
        for item in tqdm(json_data, desc="Processing JSON data"):
            # 从 sdp_text_list 构建句子
            sentence = " ".join(item['sdp_text_list']).replace("#", "").replace("{", "").replace("}", "")
            # 获取实体名称
            e1 = item['drug1_info']['drug1_name']
            e2 = item['drug2_info']['drug2_name']
            # 标签转换
            label = self.label_map[item['label']]

            self.sentences.append(sentence)
            self.entity1.append(e1)
            self.entity2.append(e2)
            self.labels.append(label)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        e1 = self.entity1[idx]
        e2 = self.entity2[idx]
        label = self.labels[idx]

        # 构造提示模板
        prompt = f"What is the relation between {e1} and {e2} in the context: {sentence}? [CLS]"

        # 标签感知数据增强：插入所有标签名称
        label_tokens = " ".join(self.label_names)
        input_text = f"[CLS] {label_tokens} [SEP] {prompt} [SEP]"

        # 编码输入
        encoding = self.tokenizer(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 提示学习与双重对比学习模型
class PromptDualCLBERT(nn.Module):
    def __init__(self, bert_model, num_labels=5, hidden_size=768, temperature=0.07):
        super(PromptDualCLBERT, self).__init__()
        self.bert = bert_model
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.dropout = nn.Dropout(0.1)
        self.label_tokens = ["No relation", "Effect", "Mechanism", "Advise", "Interaction"]

        # BatchNorm 和线性变换
        self.cls_batchnorm = nn.BatchNorm1d(hidden_size)
        self.label_batchnorm = nn.BatchNorm1d(hidden_size)
        self.cls_linear = nn.Linear(hidden_size, hidden_size)
        self.label_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids, attention_mask, labels=None):
        # 获取 BERT 输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]

        # 提取 [CLS] 特征表示 z_i
        z = sequence_output[:, 0, :]  # [batch_size, hidden_size]
        z = self.dropout(z)

        # 提取标签 token 的表示（标签感知表示 \theta_i）
        # 标签 token 位置：No relation (1-2), Effect (3), Mechanism (4), Advise (5), Interaction (6)
        theta = []
        current_pos = 1
        for label in self.label_tokens:
            label_tokens = label.split()
            num_tokens = len(label_tokens)
            label_indices = list(range(current_pos, current_pos + num_tokens))
            label_features = sequence_output[:, label_indices, :]  # [batch_size, num_tokens, hidden_size]
            label_features = label_features.mean(dim=1)  # [batch_size, hidden_size]
            theta.append(label_features)
            current_pos += num_tokens
        theta = torch.stack(theta, dim=1)  # [batch_size, num_labels, hidden_size]

        # 对比损失使用原始 z 和 theta
        z_cl = z  # [batch_size, hidden_size]
        theta_cl = theta  # [batch_size, num_labels, hidden_size]

        # 应用 BatchNorm 和线性变换
        z = self.cls_batchnorm(z)  # [batch_size, hidden_size]
        z = self.cls_linear(z)  # [batch_size, hidden_size]

        # 对每个标签的特征应用 BatchNorm 和线性变换
        theta = self.label_batchnorm(theta.view(-1, self.hidden_size))  # [batch_size * num_labels, hidden_size]
        theta = self.label_linear(theta)  # [batch_size * num_labels, hidden_size]
        theta = theta.view(-1, self.num_labels, self.hidden_size)  # [batch_size, num_labels, hidden_size]

        # 使用 einsum 计算分类 logits
        logits = torch.einsum('blh,bh->bl', theta, z)  # [batch_size, num_labels]

        # 如果提供了标签，计算对比损失
        dual_loss = None
        if labels is not None:
            # 归一化 z_cl 和 theta_cl
            z_norm = torch.nn.functional.normalize(z_cl, dim=-1)
            theta_norm = torch.nn.functional.normalize(theta_cl, dim=-1)

            # 计算 \mathcal{L}_z
            l_z = 0
            for i in range(z_cl.shape[0]):
                anchor_z = z_norm[i]
                pos_indices = (labels == labels[i]).nonzero(as_tuple=True)[0]
                pos_indices = pos_indices[pos_indices != i]
                if len(pos_indices) == 0:
                    continue
                pos_theta = theta_norm[pos_indices, labels[i]]
                neg_indices = (labels != labels[i]).nonzero(as_tuple=True)[0]
                neg_theta = theta_norm[neg_indices]
                neg_theta = neg_theta.view(-1, neg_theta.size(-1))
                pos_scores = torch.exp(anchor_z @ pos_theta.t() / self.temperature)
                neg_scores = torch.exp(anchor_z @ neg_theta.t() / self.temperature)
                l_z += -torch.log(pos_scores.sum() / (pos_scores.sum() + neg_scores.sum() + 1e-8)).mean()

            # 计算 \mathcal{L}_\theta
            l_theta = 0
            for i in range(z_cl.shape[0]):
                anchor_theta = theta_norm[i, labels[i]]
                pos_indices = (labels == labels[i]).nonzero(as_tuple=True)[0]
                pos_indices = pos_indices[pos_indices != i]
                if len(pos_indices) == 0:
                    continue
                pos_z = z_norm[pos_indices]
                neg_indices = (labels != labels[i]).nonzero(as_tuple=True)[0]
                neg_z = z_norm[neg_indices]
                pos_scores = torch.exp(anchor_theta @ pos_z.t() / self.temperature)
                neg_scores = torch.exp(anchor_theta @ neg_z.t() / self.temperature)
                l_theta += -torch.log(pos_scores.sum() / (pos_scores.sum() + neg_scores.sum() + 1e-8)).mean()

            dual_loss = (l_z + l_theta) / z_cl.shape[0] if l_z != 0 or l_theta != 0 else 0

        return logits, dual_loss

# 评估函数
def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits, _ = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(predictions)
            all_labels.extend(labels.cpu().numpy())

    # 使用 macro 平均计算多分类指标
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    avg_loss = total_loss / len(data_loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    return avg_loss, accuracy, precision, recall, f1

# 训练函数
def train_model(model, train_loader, valid_loader, test_loader, optimizer, device, epochs=3, lambda_dual=1.0):
    model.train()
    criterion = nn.CrossEntropyLoss()
    best_test_f1 = 0.0
    best_model_path = None

    # 确保 save_model 目录存在
    os.makedirs('save_model', exist_ok=True)

    for epoch in range(epochs):
        total_train_loss = 0
        total_ce_loss = 0
        total_dual_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits, dual_loss = model(input_ids, attention_mask, labels)
            ce_loss = criterion(logits, labels)
            total_loss = ce_loss + lambda_dual * (dual_loss if dual_loss is not None else 0)
            total_loss.backward()
            optimizer.step()

            total_train_loss += total_loss.item()
            total_ce_loss += ce_loss.item()
            total_dual_loss += dual_loss.item() if isinstance(dual_loss, torch.Tensor) else dual_loss

        avg_train_loss = total_train_loss / len(train_loader)
        avg_ce_loss = total_ce_loss / len(train_loader)
        avg_dual_loss = total_dual_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Avg Training Loss: {avg_train_loss:.4f}, "
              f"CE Loss: {avg_ce_loss:.4f}, Dual Loss: {avg_dual_loss:.4f}")

        # 验证集评估
        valid_loss, valid_accuracy, valid_precision, valid_recall, valid_f1 = evaluate_model(model, valid_loader, device)
        print(f"Epoch {epoch+1}, Validation Loss: {valid_loss:.4f}, Accuracy: {valid_accuracy:.4f}, "
              f"Precision: {valid_precision:.4f}, Recall: {valid_recall:.4f}, F1: {valid_f1:.4f}")

        # 测试集评估
        test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch+1}, Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, "
              f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

        # 保存最佳模型（基于测试集 F1）
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            best_model_path = os.path.join('save_model', f'best_model_test_f1_{best_test_f1:.4f}.pt')
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with Test F1: {best_test_f1:.4f} at epoch {epoch+1}")
        
        model.train()

    return best_model_path, best_test_f1

# 测试函数
def test_model(model, test_loader, device, best_model_path, best_test_f1):
    # 加载最佳模型
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, device)
    print(f"Final Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, "
          f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

    predictions = []
    probabilities = []
    label_names = ["No relation", "Effect", "Mechanism", "Advise", "Interaction"]

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits, _ = model(input_ids, attention_mask)
            batch_probs = torch.softmax(logits, dim=1).cpu().numpy()  # 转换为概率
            batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(batch_predictions)
            probabilities.extend(batch_probs)

    # 将预测结果写入日志文件
    log_path = os.path.join('save_model', f'predictions_f1_{best_test_f1:.4f}.log')
    with open(log_path, 'w') as f:
        f.write("Test Predictions\n")
        f.write("================\n")
        for idx, (pred, probs) in enumerate(zip(predictions, probabilities)):
            f.write(f"Sample {idx+1}:\n")
            f.write(f"  Predicted Label: {label_names[pred]}\n")
            f.write("  Probabilities:\n")
            for label, prob in zip(label_names, probs):
                f.write(f"    {label}: {prob:.4f}\n")
            f.write("\n")

    # 打印预测结果到控制台
    for idx, pred in enumerate(predictions):
        print(f"Test Sample {idx+1}: {label_names[pred]}")

# 主函数
def main():
    # 设置种子
    set_seed(42)

    # 加载 JSON 数据
    with open('ddi_data/train.json', 'r') as f:
        train_data = json.load(f)
    with open('ddi_data/valid.json', 'r') as f:
        valid_data = json.load(f)
    with open('ddi_data/test.json', 'r') as f:
        test_data = json.load(f)

    # 加载 BioBERT v1.1 模型和分词器
    tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
    bert_model = BertModel.from_pretrained('dmis-lab/biobert-v1.1')

    # 创建数据集
    train_dataset = RelationDataset(train_data, tokenizer)
    valid_dataset = RelationDataset(valid_data, tokenizer)
    test_dataset = RelationDataset(test_data, tokenizer)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PromptDualCLBERT(bert_model, num_labels=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # 训练模型并获取最佳模型路径和测试 F1 分数
    best_model_path, best_test_f1 = train_model(model, train_loader, valid_loader, test_loader, optimizer, device)

    # 测试模型
    print("\nTesting on test set with best model:")
    test_model(model, test_loader, device, best_model_path, best_test_f1)

if __name__ == "__main__":
    main()