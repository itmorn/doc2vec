import jieba
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pandas as pd
# https://blog.csdn.net/alip39/article/details/95891321  数据集
df = pd.read_csv(r"D:\datas\online_shopping_10_cats.csv") #cat , label , review
data = df.values
reviews = [i[-1] for i in data]
documents = []
for i in range(len(reviews)):
    try:
        review = list(jieba.cut(reviews[i]))
        documents.append(TaggedDocument(review, tags=[i]))
    except:
        print(i)
model = Doc2Vec(documents, dm=1, size=100, window=10, min_count=5)
# 保存模型
model.save('d2v.model')
print(model)

