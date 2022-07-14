import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

########################################################################
label = []
file = open('../data/kidney/testlabel.csv')#读取预测到的标签
lable_lines = file.readlines()
lable_line_0 = lable_lines[0].strip('\n').split(',')
file.close()
#转int
#lable_line_0 = [int(i) for i in lable_line_0[1:]]
for i in range(1,len(lable_line_0)):
    label.append(str(lable_line_0[i]))
#print(label)#[5, 2, 5, 2, 5....]
print(len(label)) #5729
print(label)
###########################################
predictlabel = []
file = open('../data/kidney/precisionkidney.csv')#读取预测到的标签
lable_lines = file.readlines()
lable_line_0 = lable_lines[0].strip('\n').split(',')
file.close()
#转int
#lable_line_0 = [int(i) for i in lable_line_0[1:]]
for i in range(1,len(lable_line_0)):
    predictlabel.append(str(int(lable_line_0[i])))
#print(label)#[5, 2, 5, 2, 5....]
print(len(predictlabel)) #5729
print(predictlabel)

################################################
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

y_pred = predictlabel# ['2','2','3','1','4'] # 类似的格式
y_true = label # ['0','1','2','3','4'] # 类似的格式
# 对上面进行赋值

C = confusion_matrix(y_true, y_pred, labels=['1','2','3','4','5','6','7']) # 可将'1'等替换成自己的类别，如'cat'。

plt.matshow(C, cmap=plt.cm.YlGnBu) # 根据最下面的图按自己需求更改颜色
# plt.colorbar()

for i in range(len(C)):
    for j in range(len(C)):
        plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

# plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。

plt.ylabel('True label')
plt.xlabel('Predicted label')
# plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20}) # 设置字体大小。
# plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})
# plt.xticks(range(0,5), labels=['a','b','c','d','e']) # 将x轴或y轴坐标，刻度 替换为文字/字符
# plt.yticks(range(0,5), labels=['a','b','c','d','e'])
plt.show()









