import pandas as pd
import numpy as np
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"#使用CPU
########################################
feature = []  # 列表list类型
expression = pd.read_csv('../data/blood/comprison/traindata.csv',low_memory=False)#方便获取csv中表头对应的值，但忽略line_0
print('###################expression.shape:',expression.shape)
file = open('../data/blood/comprison/traindata.csv')  # 读取训练集文件
lines = file.readlines() #读行
line_0 = lines[0].strip('\n').split(',') #line.strip('\n')移除换行符并返回列表,split(',')通过指定分隔符,对字符串进行切片
#得到
for i in range(1,len(line_0)):
    tem = list(expression[line_0[i]])
    #print('tem',tem)
    feature.append(list(tem))
    #print('list(tem)', tem)
    # print('feature length')
    # print(len(feature)) ### feature is all the training data，the number of cells
    file.close()
#print(type(feature))#list
#array_feature = np.array(feature)
feature_train = list(feature)
print("############len(feature_train):",len(feature_train))
#print("############feature_train:",feature_train)
#############################################################
label = []
file = open('../data/blood/comprison/traincelltype.csv')#读取训练集type文件
lable_lines = file.readlines()
lable_line_0 = lable_lines[0].strip('\n').split(',')
file.close()
#转int
#lable_line_0 = [int(i) for i in lable_line_0[1:]]
for i in range(1,len(lable_line_0)):
    label.append(int(lable_line_0[i]))
#print(label)#[5, 2, 5, 2, 5....]
print(len(label)) #5729
##################################
testlabel = []
file = open('../data/blood/comprison/testcelltype.csv')#读取测试集type文件
lable_lines = file.readlines()
lable_line_0 = lable_lines[0].strip('\n').split(',')
file.close()
#转int
#lable_line_0 = [int(i) for i in lable_line_0[1:]]
for i in range(1,len(lable_line_0)):
    testlabel.append(int(lable_line_0[i]))
#print(label)#[5, 2, 5, 2, 5....]
print(len(testlabel)) #5729
##################################
feature = []  # 列表list类型
testexpression = pd.read_csv('../data/blood/comprison/testdata.csv',low_memory=False)#方便获取csv中表头对应的值，但忽略line_0
print('testexpression.shape:',testexpression.shape)#(13,1286)
file = open('../data/blood/comprison/testdata.csv')  # 读取训练集文件
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
lda = LinearDiscriminantAnalysis(n_components=3)
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
testlda = LinearDiscriminantAnalysis(n_components=3)
testlda.fit(feature_test,np.array(testlabel))
lowtest=testlda.transform(feature_test)#降维后的数据
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
with open('../data/blood/comprison/trainfloat.csv','w',newline='') as csvtrain:
    writer = csv.writer(csvtrain)
    for row in feature_train:
        writer.writerow(row)
with open('../data/blood/comprison/testfloat.csv','w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in feature_test:
        writer.writerow(row)
