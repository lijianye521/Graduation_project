
from transformers import BertTokenizer
import pandas as pd
import matplotlib.pyplot as plt

# 加载BERT分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 读取数据集
df = pd.read_csv('./dataset/GCC_total.csv', encoding='latin-1')

# 使用BERT分词器对description列进行分词，并计算每个文本的token数量
# 在应用encode方法之前，先将nan值替换为空字符串
token_lengths = df['description'].astype(str).fillna('').apply(lambda x: len(tokenizer.encode(x, add_special_tokens=True)))

# 计算中位数、最大值和最小值
median_length = token_lengths.median()
max_length = token_lengths.max()
min_length = token_lengths.min()

print(f"Median token length: {median_length}")
print(f"Maximum token length: {max_length}")
print(f"Minimum token length: {min_length}")

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(token_lengths, bins=50, color='blue', alpha=0.7)
plt.title('Distribution of Token Lengths')
plt.xlabel('Token Length')
plt.ylabel('Number of Texts')
plt.show()