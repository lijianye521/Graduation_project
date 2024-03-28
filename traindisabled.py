import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLNetTokenizer, XLNetForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# 确保读取正确的CSV文件路径
df = pd.read_csv('bug_raw.csv')

# 对标签进行编码
label_encoder = LabelEncoder()
df['who_encoded'] = label_encoder.fit_transform(df['who'])

# 准备数据集
class BugReportDataset(Dataset):
    def __init__(self, descriptions, summaries, labels, tokenizer, max_len=512):
        self.descriptions = descriptions
        self.summaries = summaries
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        description = str(self.descriptions[item])
        summary = str(self.summaries[item])
        label = self.labels[item]
        data = self.tokenizer.encode_plus(
            description,
            summary,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'review_text': description + " " + summary,
            'input_ids': data['input_ids'].flatten(),
            'attention_mask': data['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 设置XLNet的tokenizer
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

# 准备训练和测试数据
X_train, X_val, y_train, y_val = train_test_split(df[['description', 'summary']], df['who_encoded'], test_size=0.1)
train_dataset = BugReportDataset(X_train['description'].values, X_train['summary'].values, y_train.values, tokenizer)
val_dataset = BugReportDataset(X_val['description'].values, X_val['summary'].values, y_val.values, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# 初始化模型
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=len(label_encoder.classes_))

# 定义优化器
optimizer = AdamW(model.parameters(), lr=2e-5)

# 以下训练和评估过程不变
