import pandas as pd
import numpy as np
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"#使用CPU
########################################
feature = []  # 列表list类型
expression = pd.read_csv('../data/pleura/traindata.csv',low_memory=False)#方便获取csv中表头对应的值，但忽略line_0
print('###################expression.shape:',expression.shape)
file = open('../data/pleura/traindata.csv')  # 读取训练集文件
lines = file.readlines() #读行
line_0 = lines[0].strip('\n').split(',') #line.strip('\n')移除换行符并返回列表,split(',')通过指定分隔符,对字符串进行切片
#得到
for i in range(1,len(line_0)):
    tem = list(expression[line_0[i]])
    feature.append(list(tem))
    file.close()
feature_train = list(feature)
print("############len(feature_train):",len(feature_train))
#############################################################
label = []
file = open('../data/pleura/trainlabel.csv')#读取训练集type文件
lable_lines = file.readlines()
lable_line_0 = lable_lines[0].strip('\n').split(',')
file.close()
#转int
for i in range(1,len(lable_line_0)):
    label.append(int(lable_line_0[i]))
print(len(label)) #5729
##################################
testlabel = []
file = open('../data/pleura/testlabel.csv')#读取测试集type文件
lable_lines = file.readlines()
lable_line_0 = lable_lines[0].strip('\n').split(',')
file.close()
#转int
for i in range(1,len(lable_line_0)):
    testlabel.append(int(lable_line_0[i]))
print(len(testlabel)) #5729
##################################
feature = []  # 列表list类型
testexpression = pd.read_csv('../data/pleura/testdata.csv',low_memory=False)#方便获取csv中表头对应的值，但忽略line_0
print('testexpression.shape:',testexpression.shape)#(13,1286)
file = open('../data/pleura/testdata.csv')  # 读取训练集文件
lines = file.readlines() #读行
line_0 = lines[0].strip('\n').split(',') #line.strip('\n')移除换行符并返回列表,split(',')通过指定分隔符,对字符串进行切片
#得到特征
for i in range(1,len(line_0)):
    tem = list(testexpression[line_0[i]])
    feature.append(tem)
print('feature length:',len(feature)) ### feature is all the training data，the number of cells
print(type(feature))#list
file.close()
#array_feature = np.array(feature)
feature_test = list(feature)
#############################################################
#lda降维
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=4)
lda.fit(feature_train,np.array(label))
lowDmat=lda.transform(feature_train)#降维后的数据
print('降维后的数据维度：',lowDmat.shape)#(15806,20)
featurelda = []
for i in range(0,len(lowDmat)):
    tem = list(lowDmat[i])
    featurelda.append(tem)
    print(i)
    print('################type(tem)',type(tem))#list
    #print('###############tem:',tem)
#testlda = LinearDiscriminantAnalysis(n_components=4)
#lda.fit(feature_test,np.array(testlabel))
lowtest=lda.transform(feature_test)#降维后的数据
print('降维后的测试数据维度：',lowtest.shape)#(5729,800)
featuretest = []
for i in range(0,len(lowtest)):
    tem = list(lowtest[i])
    featuretest.append(tem)
    print(i)
    print('################testtype(tem)',type(tem))#list

feature_train = list(featurelda)
feature_test = list(featuretest)
length =len(feature_train[0])
print('###########len(feature_train[0]):',length)
print('len(feature_train):',len(feature_train))
print('%%%%%%%%%feature_train[0]:',feature_train[0])
import csv
with open('../data/pleura/trainfloat.csv','w',newline='') as csvtrain:
    writer = csv.writer(csvtrain)
    for row in feature_train:
        writer.writerow(row)
with open('../data/pleura/testfloat.csv','w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in feature_test:
        writer.writerow(row)
