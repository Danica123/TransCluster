import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##########################################################
class DrawConfusionMatrix:
    def __init__(self, labels_name, normalize=True):
        """
		normalize：是否设元素为百分比形式
        """
        self.normalize = normalize
        self.labels_name = labels_name
        self.num_classes = len(labels_name)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype="float32")

    def update(self, predicts, labels):
        """

        :param predicts: 一维预测向量，eg：array([0,5,1,6,3,...],dtype=int64)
        :param labels:   一维标签向量：eg：array([0,5,0,6,2,...],dtype=int64)
        :return:
        """
        for predict, label in zip(predicts, labels):
            self.matrix[predict, label] += 1

    def getMatrix(self,normalize=True):
        """
        根据传入的normalize判断要进行percent的转换，
        如果normalize为True，则矩阵元素转换为百分比形式，
        如果normalize为False，则矩阵元素就为数量
        Returns:返回一个以百分比或者数量为元素的矩阵

        """
        if normalize:
            per_sum = self.matrix.sum(axis=1)  # 计算每行的和，用于百分比计算
            for i in range(self.num_classes):
                self.matrix[i] =(self.matrix[i] / per_sum[i])   # 百分比转换
            self.matrix=np.around(self.matrix, 2)   # 保留2位小数点
            self.matrix[np.isnan(self.matrix)] = 0  # 可能存在NaN，将其设为0
        return self.matrix

    def drawMatrix(self):
        self.matrix = self.getMatrix(self.normalize)
        plt.imshow(self.matrix, cmap=plt.cm.Blues)  # 仅画出颜色格子，没有值
        plt.title("Normalized confusion matrix")  # title
        plt.xlabel("Predict label")
        plt.ylabel("Truth label")
        plt.yticks(range(self.num_classes), self.labels_name)  # y轴标签
        plt.xticks(range(self.num_classes), self.labels_name, rotation=45)  # x轴标签

        for x in range(self.num_classes):
            for y in range(self.num_classes):
                value = float(format('%.2f' % self.matrix[y, x]))  # 数值处理
                plt.text(x, y, value, verticalalignment='center', horizontalalignment='center')  # 写值

        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

        plt.colorbar()  # 色条
        plt.savefig('./ConfusionMatrix.png', bbox_inches='tight')  # bbox_inches='tight'可确保标签信息显示全
        plt.show()

########################################################################
label = []
file = open('../data/kidney/testlabel.csv')#读取预测到的标签
lable_lines = file.readlines()
lable_line_0 = lable_lines[0].strip('\n').split(',')
file.close()
#转int
#lable_line_0 = [int(i) for i in lable_line_0[1:]]
for i in range(1,len(lable_line_0)):
    label.append(int(lable_line_0[i]))
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
    predictlabel.append(int(lable_line_0[i]))
#print(label)#[5, 2, 5, 2, 5....]
print(len(predictlabel)) #5729
print(predictlabel)

################################################
labels_name = ['1', '2', '3', '4', '5', '6', '7']
drawconfusionmatrix = DrawConfusionMatrix(labels_name = labels_name)  #实例化

for index, (labels, imgs) in enumerate(test_loader):
    labels_np = label
    predict_np = predictlabel  # array([0,5,1,6,3,...],dtype=int64)
    drawconfusionmatrix.update(predict_np, labels_np)  # 将新批次的predict和label更新（保存）
# labels_np = label
# predict_np = predictlabel  # array([0,5,1,6,3,...],dtype=int64)
# drawconfusionmatrix.update(predict_np, labels_np)  # 将新批次的predict和label更新（保存）

drawconfusionmatrix.drawMatrix()  # 根据所有predict和label，画出混淆矩阵

# confusion_mat = drawconfusionmatrix.getMatrix()  # 你也可以使用该函数获取混淆矩阵(ndarray)
# print(confusion)








