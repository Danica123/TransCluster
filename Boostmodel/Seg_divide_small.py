# 把原始的Seg数据集划分为70%训练集，15%测试集，15%验证集，去掉不要的
# 输出来的数据集最后面会有一列 nan
# 运行后直接输出三个csv到指定位置，每次运行得到dataset不同（打乱顺序随机）
# num_train,num_test,num_valid 分别为每个数据集里数据条数，和为data_all的总条数。

import numpy as np
import pandas as pd
# 输入文件
file = []
file.append('/home/aita/4444/jiao/cellclassification/mydata/Segerstolpe/Segerstolpe.csv')
cell_head = []
gene_head = []
head = []
data = pd.read_csv(file[0], header=None,low_memory=False)
gene_head = data[0]
data = np.array(data)
data = data[:,1:]
cell_head = data[0]
data = data[:,1:]
head = data[0]
data = data[1:,:]
data = data.transpose()
gene_head = np.array(gene_head)
temp2 = []
temp2.append('cellname')
for i in range(len(gene_head)):
    temp2.append(gene_head[i])
gene_head = temp2
cell_head = cell_head[1:]



file2 = []
file2.append('/home/aita/4444/jiao/cellclassification/mydata/Segerstolpe/bridge.csv')
bridge = pd.read_csv(file2[0], header=None,low_memory=False)
bridge = np.array(bridge)

file3 = []
file3.append('/home/aita/4444/jiao/cellclassification/mydata/Segerstolpe/type.csv')
type = pd.read_csv(file3[0], low_memory=False)
type = np.array(type)
type = type.transpose()

type_all = []
data_all = []
cell_head_all = []
for i in range(len(head)):
    for j in range(type.shape[1]):
        if head[i] == type[0][j]:
            type_all.append(type[1][j])
            break
type_all = np.array(type_all)

temp = []
for i in range(len(type_all)):
    for j in range(bridge.shape[1]):
        if type_all[i] == bridge[0][j]:
            temp.append(bridge[0][j])
            data_all.append(data[i])
            cell_head_all.append(cell_head[i])
type_all = np.array(temp)
data_all = np.array(data_all)
cell_head_all = np.array(cell_head_all)

# 打乱顺序
index = [i for i in range(data_all.shape[0])]
np.random.shuffle(index)
data_all = data_all[index]
type_all = type_all[index]
cell_head_all = cell_head_all[index]
data_all = np.array(data_all)
type_all = np.array(type_all)
cell_head_all = np.array(cell_head_all)
print('打乱后',type_all.shape)
print('打乱后',data_all.shape)
print('打乱后',cell_head_all.shape)



print('data_all',data_all)
gene_head = np.array(gene_head)
print('gene_head',gene_head.shape)
print('gene_head',gene_head)
# 删除基因都为0的基因
data_all = data_all.transpose()
print('转置data_all',data_all.shape)
# print('转置data_all',data_all)
# print('gene_head',gene_head)
gene_head = gene_head[2:]
print('删除无用元素gene_head',gene_head.shape)
data_all_small = []
gene_head_small = []
for i in range(data_all.shape[0]):
    temp = 0
    for j in range(data_all.shape[1]):
        temp = float(data_all[i][j]) + temp
    # if temp == 0:
    #     print('删掉基因',i)
    if temp != 0:
        data_all_small.append(data_all[i])
        gene_head_small.append(gene_head[i])
data_all_small = np.array(data_all_small)
data_all_small = data_all_small.transpose()
data_all = data_all_small
gene_head = np.array(gene_head_small)
print('删除表达量都为0的基因后：',data_all.shape)
print('删除表达量都为0的基因后：',gene_head.shape)
temp2 = []
temp2.append('cellname')
temp2.append('celltype')
for i in range(len(gene_head)):
    temp2.append(gene_head[i])
gene_head = np.array(temp2)
#
#
#
# 输出csv
num_train = 1448 # 打印个数
with open("/home/aita/4444/jiao/cellclassification/mydata/Segerstolpe/train_small.csv", "w") as file_train:
    print("开始写train,数据条数：",num_train)
    for i in range(len(gene_head)):
        file_train.write(str(gene_head[i]))
        file_train.write(',')
    file_train.write("\n")
    for i in range(num_train):
        file_train.write(str(cell_head_all[i]))
        file_train.write(',')
        file_train.write(str(type_all[i]))
        file_train.write(',')
        for j in range(data_all.shape[1]):
            file_train.write(str(data_all[i][j]))
            file_train.write(',')
        file_train.write("\n")

num_test = 310 # 打印个数
with open("/home/aita/4444/jiao/cellclassification/mydata/Segerstolpe/test_small.csv", "w") as file_test:
    print("开始写test,数据条数：",num_test)
    for i in range(len(gene_head)):
        file_test.write(str(gene_head[i]))
        file_test.write(',')
    file_test.write("\n")
    for i in range(num_train,(num_train + num_test)):
        file_test.write(str(cell_head_all[i]))
        file_test.write(',')
        file_test.write(str(type_all[i]))
        file_test.write(',')
        for j in range(data_all.shape[1]):
            file_test.write(str(data_all[i][j]))
            file_test.write(',')
        file_test.write("\n")

num_valid = 310 # 打印个数
with open("/home/aita/4444/jiao/cellclassification/mydata/Segerstolpe/valid_small.csv", "w") as file_valid:
    print("开始写valid,数据条数：",num_valid)
    for i in range(len(gene_head)):
        file_valid.write(str(gene_head[i]))
        file_valid.write(',')
    file_valid.write("\n")
    for i in range((num_train + num_test),(num_train + num_test + num_valid)):
        file_valid.write(str(cell_head_all[i]))
        file_valid.write(',')
        file_valid.write(str(type_all[i]))
        file_valid.write(',')
        for j in range(data_all.shape[1]):
            file_valid.write(str(data_all[i][j]))
            file_valid.write(',')
        file_valid.write("\n")
