import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import numpy as np


# 自定义数据集类
class RelationDataset(Dataset):
    def __init__(self, sentences, entity1, entity2, labels, tokenizer, max_length=128):
        self.sentences = sentences
        self.entity1 = entity1
        self.entity2 = entity2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

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
        cls_output = outputs.last_hidden_state[:, 0, :]  # 取[CLS] token的输出
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits


# 训练函数
def train_model(model, train_loader, optimizer, device, epochs=3):
    model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(train_loader)}")


# 主函数
def main():
    # 加载预训练BERT模型和分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    # 示例数据
    sentences = [
        "John works with Mary in the same company.",
        "Alice and Bob are unrelated.",
        "Tom is the manager of Sarah."
    ]
    entity1 = ["John", "Alice", "Tom"]
    entity2 = ["Mary", "Bob", "Sarah"]
    labels = [1, 0, 1]  # 1表示有关系，0表示无关系

    # 创建数据集
    dataset = RelationDataset(sentences, entity1, entity2, labels, tokenizer)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PromptBERT(bert_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # 训练模型
    train_model(model, train_loader, optimizer, device)

    # 推理示例
    model.eval()
    test_sentence = "John works with Mary in the same company."
    test_prompt = f"{test_sentence} Does John have a relation with Mary? [CLS]"
    encoding = tokenizer(
        test_prompt,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        prediction = torch.argmax(logits, dim=1).item()

    print(f"Prediction: {'Relation exists' if prediction == 1 else 'No relation'}")


if __name__ == "__main__":
    main()