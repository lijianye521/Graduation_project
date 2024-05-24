import torch
from transformers import XLNetTokenizer, XLNetForSequenceClassification
import numpy as np
import pandas as pd

# 定义模型和tokenizer的路径
MODEL_PATH = 'model_checkpoint_xlnet_top1-top10_eclipse_dataprocessed4444444OpenOffice_total_10_10.pth'
TOKENIZER_PATH = 'xlnet-base-cased'
CSV_FILE_PATH = './dataset2/OpenOffice_total_10_10.csv'

# 指定需要提取的列
columns_to_extract = ['developer']
df = pd.read_csv(CSV_FILE_PATH, usecols=columns_to_extract, encoding='latin-1')

# 将developer列作为标签
label_dict = {label: idx for idx, label in enumerate(df['developer'].unique())}
reverse_label_dict = {v: k for k, v in label_dict.items()}  # 反向字典用于从索引获取开发者名称

# 加载模型和tokenizer
tokenizer = XLNetTokenizer.from_pretrained(TOKENIZER_PATH)
model = XLNetForSequenceClassification.from_pretrained(TOKENIZER_PATH, num_labels=len(label_dict))

# 加载训练好的模型参数
checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_developers(text):
    # 对输入文本进行编码
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=512
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs[0]
    probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # 获取前10个开发者及其概率
    topk_indices = np.argsort(probabilities)[-10:][::-1]
    topk_probabilities = probabilities[topk_indices]

    results = [(reverse_label_dict[idx], prob) for idx, prob in zip(topk_indices, topk_probabilities)]
    
    return results

# 示例输入文本
input_text = "hu translat duplic menu element localisation ,translat duplic place officecfg registry data org openoffice office xcu commands uno label officecfg registry data org openoffice office xcu commands uno label,ui,trivial,"

# 获取预测结果
predicted_developers = predict_developers(input_text)

# 打印预测结果
for developer, probability in predicted_developers:
    print(f"Developer: {developer}, Probability: {probability:.4f}")
