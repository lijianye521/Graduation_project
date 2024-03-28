import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLNetTokenizer, XLNetForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# 加载新的数据集
new_file_path = './data_开发者阈值10_词频阈值10/Mozilla_total_10_10.csv'
columns_to_extract = ['bug_id', 'product', 'abstracts', 'description', 'component', 'severity', 'priority', 'history', 'status', 'developer']
df = pd.read_csv(new_file_path, usecols=columns_to_extract, encoding='latin-1')

# 合并文本信息为模型的输入，除了developer列
df['text_input'] = df[['bug_id', 'product', 'abstracts', 'description', 'component', 'severity', 'priority', 'history', 'status']].astype(str).agg(' '.join, axis=1)

# 将developer列作为标签
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['developer'])

# 准备数据集
class BugReportDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        data = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids': data['input_ids'].flatten(),
            'attention_mask': data['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 设置XLNet的tokenizer
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

# 准备训练和测试数据
X_train, X_val, y_train, y_val = train_test_split(df['text_input'], df['label'], test_size=0.1, random_state=42, stratify=df['label'])
train_dataset = BugReportDataset(X_train.values, y_train.values, tokenizer)
val_dataset = BugReportDataset(X_val.values, y_val.values, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# 初始化XLNet模型
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=len(label_encoder.classes_))

# 定义优化器
optimizer = AdamW(model.parameters(), lr=2e-5)

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# 简单评估
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print(f'Accuracy: {100 * correct / total}')