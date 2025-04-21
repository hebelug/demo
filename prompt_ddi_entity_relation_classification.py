import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np

# 自定义数据集类
class RelationDataset(Dataset):
    def __init__(self, json_data, tokenizer, max_length=128):
        self.sentences = []
        self.entity1 = []
        self.entity2 = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 处理 JSON 数据
        for item in json_data:
            # 从 sdp_text_list 构建句子
            sentence = " ".join(item['sdp_text_list']).replace("#", "").replace("{", "").replace("}", "")
            # 获取实体名称
            e1 = item['drug1_info']['drug1_name']
            e2 = item['drug2_info']['drug2_name']
            # 标签转换：NA 为 0，其他为 1（可根据实际标签调整）
            label = 0 if item['label'] == "NA" else 1

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
        prompt = f"{sentence} Does {e1} have a relation with {e2}? [CLS]"

        # 编码输入
        encoding = self.tokenizer(
            prompt,
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

# 提示学习模型
class PromptBERT(nn.Module):
    def __init__(self, bert_model, num_labels=2):
        super(PromptBERT, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs[0][:, 0, :]  # 取元组的第一个元素（last_hidden_state），并选择 [CLS] token
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits

# 训练函数
def train_model(model, train_loader, valid_loader, optimizer, device, epochs=3):
    model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_train_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        print(f"Epoch {epoch+1}, Average Training Loss: {total_train_loss / len(train_loader)}")

        # 验证集评估
        model.eval()
        total_valid_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                total_valid_loss += loss.item()

                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Epoch {epoch+1}, Average Validation Loss: {total_valid_loss / len(valid_loader)}, Validation Accuracy: {accuracy}")
        model.train()

# 测试函数
def test_model(model, test_loader, tokenizer, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits = model(input_ids, attention_mask)
            batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(batch_predictions)

    # 打印测试结果
    for idx, pred in enumerate(predictions):
        print(f"Test Sample {idx+1}: {'Relation exists' if pred == 1 else 'No relation'}")

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
    model = PromptBERT(bert_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # 训练模型
    train_model(model, train_loader, valid_loader, optimizer, device)

    # 测试模型
    test_model(model, test_loader, tokenizer, device)

if __name__ == "__main__":
    main()