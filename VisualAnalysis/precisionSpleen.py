import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris,load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import umap

feature = []  # 列表list类型
testexpression = pd.read_csv('../data/Spleen/testdata.csv')#方便获取csv中表头对应的值，但忽略line_0
print('testexpression.shape:',testexpression.shape)#(16382,1286)1286条记录，每条记录16382个特征
file = open('../data/Spleen/testdata.csv')  # 读取训练集文件
lines = file.readlines() #读行
line_0 = lines[0].strip('\n').split(',') #line.strip('\n')移除换行符并返回列表,split(',')通过指定分隔符,对字符串进行切片
#得到特征
for i in range(1,len(line_0)):
    tem = list(testexpression[line_0[i]])
    feature.append(tem)
print('feature length:',len(feature)) ### feature is all the training data，the number of cells1285
print(type(feature))#list
file.close()
array_feature = np.array(feature)
print(array_feature)
###########################################
testlabel = []
file = open('../data/Spleen/precisionspleencsv.csv')#读取测试集type文件
lable_lines = file.readlines()
lable_line_0 = lable_lines[0].strip('\n').split(',')
file.close()
#转int
#lable_line_0 = [int(i) for i in lable_line_0[1:]]
for i in range(1,len(lable_line_0)):
    testlabel.append(int(lable_line_0[i]))
#print(label)#[5, 2, 5, 2, 5....]
print(len(testlabel)) #5729
print(testlabel)
################################################
X_tsne = TSNE(n_components=2,random_state=100).fit_transform(array_feature)
pca = PCA(n_components=2).fit_transform(array_feature)
################################################
reducer = umap.UMAP(random_state=42)
reducer.fit(array_feature)
embedding = reducer.transform(array_feature)
assert(np.all(embedding == reducer.embedding_))
embedding.shape

plt.figure(figsize=(15,10))
plt.scatter(embedding[:,0],embedding[:,1],c=testlabel,cmap='Spectral',s=5)
plt.gca().set_aspect('equal','datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the Digits dataset', fontsize=24);

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=testlabel,label="t-SNE")
plt.legend()
plt.subplot(122)
plt.scatter(pca[:,0],pca[:,1],c=testlabel,label="PCA")
plt.legend()
plt.colorbar()

plt.savefig('../pictures/digits_tsne.png', dpi=120)
plt.show()


