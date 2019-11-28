import jieba
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pandas as pd
model = Doc2Vec.load('d2v.model')

df = pd.read_csv(r"D:\datas\online_shopping_10_cats.csv") #cat , label , review
data = df.values
reviews = [[i[-1],i[-2]] for i in data if len(str(i[-1]))>10]
documents = []
count = 0
for review in reviews:
    if count%1999==0:
        print(count)
    count+=1
    a = list(jieba.cut(review[0]))
    documents.append([model.infer_vector(a),review[1]])

X = [i[0] for i in documents]
Y = [i[1] for i in documents]
print(Y)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train,y_train)
a = tree_model.score(X_test,y_test)
print(a)
# 0.6466596206263558

from lightgbm.sklearn import LGBMClassifier
tree_model = LGBMClassifier(n_estimators=500)
tree_model.fit(X_train,y_train)
a = tree_model.score(X_test,y_test)
print(tree_model)
print(a)
# 0.8273905545975413