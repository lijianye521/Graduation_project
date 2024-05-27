#test 不用管
from transformers import GPT2Tokenizer, GPT2Model

# 加载分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 设置pad_token为eos_token，因为GPT-2默认没有pad_token
tokenizer.pad_token = tokenizer.eos_token

# 示例文本
texts = ["这是一个例子。", "这是一个更长的例子文本。"]

# 使用分词器处理文本，启用填充
inputs = tokenizer(texts, padding=True, return_tensors="pt")

# inputs现在包含了填充后的input_ids和attention_mask，可以直接用于模型
print(inputs)
import os
import torch
from tqdm.auto import tqdm
import warnings
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import pandas as pd


# 忽略特定的警告
warnings.filterwarnings("ignore", message="Be aware, overflowing tokens are not returned*")

new_file_path = './dataset2/OpenOffice_total_10_10.csv'
# 指定需要提取的列
columns_to_extract = ['bug_id', 'product', 'abstracts', 'description', 'component', 'severity', 'priority', 'developer',  'status']
# columns_to_extract = [ 'description', 'developer']
df = pd.read_csv(new_file_path, usecols=columns_to_extract, encoding='latin-1')
# 将developer列作为标签
label_dict = {label: idx for idx, label in enumerate(df['developer'].unique())}
print(f' the number of label is {len(label_dict)}')
df['label'] = df['developer'].replace(label_dict).infer_objects()
# 合并bug_id和summary作为模型的输入
df['text_input'] = df['abstracts'].astype(str) + " " + df['description'].astype(str)  # 使用空格作为分隔符
X_train, X_val, y_train, y_val = train_test_split(df.index.values, df.label.values, test_size=0.15, random_state=42, stratify=df.label.values)
df['data_type'] = ['not_set']*df.shape[0]
df.loc[X_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'
# 对训练和验证数据的合并文本进行编码
encoded_data_train = tokenizer.batch_encode_plus(
    df[df.data_type=='train'].text_input.values,  # 使用合并后的文本
    add_special_tokens=True, 
    return_attention_mask=True, 
    padding='max_length',  # 更新pad_to_max_length为padding
    max_length=512, 
    truncation=True,  # 明确启用截断
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    df[df.data_type=='val'].text_input.values,  # 使用合并后的文本
    add_special_tokens=True, 
    return_attention_mask=True, 
    padding='max_length',  # 更新pad_to_max_length为padding
    max_length=512, 
    truncation=True,  # 明确启用截断
    return_tensors='pt'
)

# 准备Tensor数据
input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df[df.data_type=='train'].label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type=='val'].label.values)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

# 定义DataLoader
batch_size = 2
train_loader = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size)
val_loader = DataLoader(dataset_val, sampler=SequentialSampler(dataset_val), batch_size=2)
import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2Config, AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 确保这里使用的是正确的分词器变量名
GPT2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# 设置pad_token为eos_token
GPT2_tokenizer.pad_token = GPT2_tokenizer.eos_token

# 通过GPT2Config设置num_labels参数
num_labels = len(label_dict)  # 请替换为实际的标签数量
configuration = GPT2Config.from_pretrained("gpt2", num_labels=num_labels)

model = GPT2ForSequenceClassification.from_pretrained("gpt2", config=configuration).to(device)
# 设置模型的pad_token_id
#感谢stackoverflow
model.config.pad_token_id = GPT2_tokenizer.pad_token_id
# 使用GPT2_tokenizer的长度来调整模型的token embeddings大小
model.resize_token_embeddings(len(GPT2_tokenizer))

optimizer = AdamW(model.parameters(), lr=1e-5, eps=1e-8)

model.to(device)

checkpoint_path = 'model_xuezhang——ne44t.pth'

# 检查是否有可用的检查点
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f'Resuming training from epoch {start_epoch}')
else:
    start_epoch = 0
    print('Starting training from scratch')

import pymysql
from datetime import datetime

# 数据库连接信息
host = '38.147.173.234'
user = 'lijianye'
password = '660013'
db = 'training_statistics_db'
experiment_num = 13
# 模型名称，根据实际情况手动设置
model_name = 'gpt2'
# 学习率和可选特性，根据实际情况手动设置
learning_rate = 1e-5  # 示例学习率
optional_feature = 'abstract+descrition'  # 示例可选特性
dataset = new_file_path
num_epochs = 15
for epoch in range(start_epoch, num_epochs):
    model.train()
    start_time = datetime.now()
        
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=loss.item())
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, checkpoint_path)
    model.eval()
    correct_topk = {k: 0 for k in range(1, 11)}
    total = 0
    val_progress_bar = tqdm(val_loader, desc="Validating")
    
    for batch in val_progress_bar:
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs[0]
        total += labels.size(0)
        
        # 计算top1到top10的正确率
        _, predicted_topk = torch.topk(logits, k=10, dim=1)
        labels_expanded = labels.unsqueeze(1)
        for k in range(1, 11):
            correct_topk[k] += (predicted_topk[:, :k] == labels_expanded).any(dim=1).sum().item()
                
    # 打印每个topK的准确率
    top10accuracy = []  # 初始化存储Top1到Top10准确率的数组

    for k in range(1, 11):
        accuracy = 100 * correct_topk[k] / total
        top10accuracy.append(accuracy)  # 将计算出的准确率添加到数组中
        print(f'Accuracy after epoch {epoch + 1}: Top{k}: {accuracy:.2f}%')
        print(top10accuracy)
    import pandas as pd
    import os
        # ...训练结束时间
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()/60.0
    # 定义数据字典，用于创建DataFrame
    data = {
            'epoch': [epoch],
            'start_time': [start_time],
            'end_time': [end_time],
            'duration': [duration],
            'user_id': [1],
            'model': [model_name],
            'top1_accuracy': [top10accuracy[0]],
            'top2_accuracy': [top10accuracy[1]],
            'top3_accuracy': [top10accuracy[2]],
            'top4_accuracy': [top10accuracy[3]],
            'top5_accuracy': [top10accuracy[4]],
            'top6_accuracy': [top10accuracy[5]],
            'top7_accuracy': [top10accuracy[6]],
            'top8_accuracy': [top10accuracy[7]],
            'top9_accuracy': [top10accuracy[8]],
            'top10_accuracy': [top10accuracy[9]],
            'optional_feature': [optional_feature],
            'learning_rate': [learning_rate],
            'dataset': [dataset],
            'experiment_num':[experiment_num],
    }

        # 创建DataFrame
    df = pd.DataFrame(data)

        # 检查train.csv文件是否存在来决定是否添加表头
    file_exists = os.path.isfile('modified_train_with_updated_duration.csv')

        # 如果文件存在，不写入表头，模式为追加；如果文件不存在，写入表头，模式为写入
    df.to_csv('xuezhangtrainoutcome.csv', mode='a', header=not file_exists, index=False)

    print(f'Epoch {epoch + 1} training data inserted into train.csv.')