import pandas as pd
probability = open('../modelsave/bloodlarge_lda_transformer_float_epoch20.txt')
lines = probability.readlines()
llll = []
for i in lines:
    temp = i.strip('\n')
    temp_list = temp.split(',')
    t = []
    for j in range(1,len(temp_list)-1):
        t.append(float(temp_list[j]))
    llll.append(t)
print(llll)
labelssss = []
for i in llll:
    labelssss.append(i.index(max(i))+1)
print(labelssss)#预测出来的标签
print(len(labelssss))
probability.close()

label44 = open('../data/blood/3223label.csv')
lines44 = label44.readlines()
a = lines44[0].strip('\n').split(',')
print(len(a))#真实标签的长度
jjjjj= []
for i in range(1,len(a)):
    jjjjj.append(int(a[i]))
print(jjjjj)#真实标签
count = 0
#计算accuracy
for i in range(len(jjjjj)):
    if (jjjjj[i]==labelssss[i]):
        count +=1
print('accuracy:',count/len(jjjjj))
#计算precision，计算recall，计算f1-score
from sklearn.metrics import f1_score,precision_score,recall_score
f1 = f1_score(y_true=jjjjj,y_pred=labelssss,average='macro')
precision = precision_score(y_true=jjjjj,y_pred=labelssss,average='macro')
recall = recall_score(y_true=jjjjj,y_pred=labelssss,average='macro')
print('f1:',f1)
print('precision:',precision)
print('recall:',recall)