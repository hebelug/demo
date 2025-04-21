import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

# 自定义数据集类
class RelationDataset(Dataset):
    def __init__(self, json_data, tokenizer, max_length=128):
        self.sentences = []
        self.entity1 = []
        self.entity2 = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_names = ["No relation", "Effect", "Mechanism", "Advise", "Interaction"]  # 更新标签名称
        self.label_map = {"NA": 0, "effect": 1, "mechanism": 2, "advise": 3, "int": 4}  # 更新标签映射

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
    def __init__(self, bert_model, num_labels=5, temperature=0.07):  # 更新为5个标签
        super(PromptDualCLBERT, self).__init__()
        self.bert = bert_model
        self.num_labels = num_labels
        self.temperature = temperature
        self.dropout = nn.Dropout(0.1)
        self.label_tokens = ["No relation", "Effect", "Mechanism", "Advise", "Interaction"]  # 更新标签名称

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
        current_pos = 1  # 从 [CLS] 后的第一个 token 开始
        for label in self.label_tokens:
            label_tokens = label.split()
            num_tokens = len(label_tokens)
            label_indices = list(range(current_pos, current_pos + num_tokens))  # 标签 token 位置
            label_features = sequence_output[:, label_indices, :]  # [batch_size, num_tokens, hidden_size]
            label_features = label_features.mean(dim=1)  # [batch_size, hidden_size]
            theta.append(label_features)
            current_pos += num_tokens
        theta = torch.stack(theta, dim=1)  # [batch_size, num_labels, hidden_size]

        # 计算分类 logits
        logits = torch.bmm(theta, z.unsqueeze(-1)).squeeze(-1)  # [batch_size, num_labels]

        # 如果提供了标签，计算对比损失
        dual_loss = None
        if labels is not None:
            # 归一化 z 和 theta
            z_norm = torch.nn.functional.normalize(z, dim=-1)
            theta_norm = torch.nn.functional.normalize(theta, dim=-1)

            # 计算 \mathcal{L}_z
            l_z = 0
            for i in range(z.shape[0]):  # 遍历每个样本
                anchor_z = z_norm[i]  # [hidden_size]
                pos_indices = (labels == labels[i]).nonzero(as_tuple=True)[0]
                pos_indices = pos_indices[pos_indices != i]  # 排除自身
                if len(pos_indices) == 0:
                    continue  # 跳过没有正样本的情况
                pos_theta = theta_norm[pos_indices, labels[i]]  # 正样本的 \theta^*
                neg_indices = (labels != labels[i]).nonzero(as_tuple=True)[0]
                neg_theta = theta_norm[neg_indices]  # 负样本的 \theta^*
                neg_theta = neg_theta.view(-1, neg_theta.size(-1))  # 展平负样本维度
                pos_scores = torch.exp(anchor_z @ pos_theta.t() / self.temperature)  # [1, |P_i|]
                neg_scores = torch.exp(anchor_z @ neg_theta.t() / self.temperature)  # [1, |A_i \ P_i|]
                l_z += -torch.log(pos_scores.sum() / (pos_scores.sum() + neg_scores.sum() + 1e-8)).mean()

            # 计算 \mathcal{L}_\theta
            l_theta = 0
            for i in range(z.shape[0]):
                anchor_theta = theta_norm[i, labels[i]]  # [hidden_size]
                pos_indices = (labels == labels[i]).nonzero(as_tuple=True)[0]
                pos_indices = pos_indices[pos_indices != i]
                if len(pos_indices) == 0:
                    continue
                pos_z = z_norm[pos_indices]  # 正样本的 z
                neg_indices = (labels != labels[i]).nonzero(as_tuple=True)[0]
                neg_z = z_norm[neg_indices]  # 负样本的 z
                pos_scores = torch.exp(anchor_theta @ pos_z.t() / self.temperature)
                neg_scores = torch.exp(anchor_theta @ neg_z.t() / self.temperature)
                l_theta += -torch.log(pos_scores.sum() / (pos_scores.sum() + neg_scores.sum() + 1e-8)).mean()

            dual_loss = (l_z + l_theta) / z.shape[0] if l_z != 0 or l_theta != 0 else 0

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
def train_model(model, train_loader, valid_loader, optimizer, device, epochs=3, lambda_dual=1.0):
    model.train()
    criterion = nn.CrossEntropyLoss()

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
            total_dual_loss += dual_loss.item() if dual_loss is not None else 0

        avg_train_loss = total_train_loss / len(train_loader)
        avg_ce_loss = total_ce_loss / len(train_loader)
        avg_dual_loss = total_dual_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Avg Training Loss: {avg_train_loss:.4f}, "
              f"CE Loss: {avg_ce_loss:.4f}, Dual Loss: {avg_dual_loss:.4f}")

        # 验证集评估
        valid_loss, valid_accuracy, valid_precision, valid_recall, valid_f1 = evaluate_model(model, valid_loader, device)
        print(f"Epoch {epoch+1}, Validation Loss: {valid_loss:.4f}, Accuracy: {valid_accuracy:.4f}, "
              f"Precision: {valid_precision:.4f}, Recall: {valid_recall:.4f}, F1: {valid_f1:.4f}")
        
        model.train()

# 测试函数
def test_model(model, test_loader, device):
    test_loss, test_accuracy, test_precision, test_recall, test_f1 = evaluate_model(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, "
          f"Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

    model.eval()
    predictions = []
    label_names = ["No relation", "Effect", "Mechanism", "Advise", "Interaction"]  # 更新标签名称
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits, _ = model(input_ids, attention_mask)
            batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(batch_predictions)

    for idx, pred in enumerate(predictions):
        print(f"Test Sample {idx+1}: {label_names[pred]}")

# 主函数
def main():
    # 加载 JSON 数据
    with open('train.json', 'r') as f:
        train_data = json.load(f)
    with open('valid.json', 'r') as f:
        valid_data = json.load(f)
    with open('test.json', 'r') as f:
        test_data = json.load(f)

    # 加载预训练BERT模型和分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    # 创建数据集
    train_dataset = RelationDataset(train_data, tokenizer)
    valid_dataset = RelationDataset(valid_data, tokenizer)
    test_dataset = RelationDataset(test_data, tokenizer)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PromptDualCLBERT(bert_model, num_labels=5).to(device)  # 更新为5个标签
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # 训练模型
    train_model(model, train_loader, valid_loader, optimizer, device)

    # 测试模型
    print("\nTesting on test set:")
    test_model(model, test_loader, device)

if __name__ == "__main__":
    main()