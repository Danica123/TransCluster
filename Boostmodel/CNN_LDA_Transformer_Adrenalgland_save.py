import pandas as pd
from tensorflow import keras
from transformer import Transformer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Conv1D,MaxPooling1D,Dense,Dropout,Flatten
from tensorflow.python.keras.layers import Activation, SpatialDropout1D, Convolution1D, GlobalMaxPooling1D
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"#使用CPU
########################################
feature = []  # 列表list类型
expression = pd.read_csv('../data/Adrenal gland/glandtrain1.csv')#方便获取csv中表头对应的值，但忽略line_0
print('###################expression.shape:',expression.shape)
file = open('../data/Adrenal gland/glandtrain1.csv')  # 读取训练集文件
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
file = open('../data/Adrenal gland/traincelltype.csv')#读取训练集文件
lable_lines = file.readlines()
lable_line_0 = lable_lines[0].strip('\n').split(',')
file.close()

#转int
#lable_line_0 = [int(i) for i in lable_line_0[1:]]
for i in range(1,len(lable_line_0)):
    label.append(int(lable_line_0[i]))
#print(label)#[5, 2, 5, 2, 5....]
print(len(label)) #5729
#将特征标签转换成[0,0,0,0,1,0]格式
y_train=[]
for i in label:
    tem =[]
    for j in range(0,9):
        tem.append(0)
    tem[i-1]=1
    y_train.append(tem)
#print('label length')           ##########training data label
print(len(y_train))  #5729
#print('y_train:',y_train)

#######generate the test data and test data lebels
##################################
feature = []  # 列表list类型
testexpression = pd.read_csv('../data/Adrenal gland/glandtest1.csv')#方便获取csv中表头对应的值，但忽略line_0
print('testexpression.shape:',testexpression.shape)#(13,1286)
file = open('../data/Adrenal gland/glandtest1.csv')  # 读取训练集文件
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

##################################
#lda降维
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# lda = LinearDiscriminantAnalysis(n_components=13)
# lda.fit(feature_train,np.array(label))
# lowDmat=lda.transform(feature_train)#降维后的数据
# print('降维后的数据维度：',lowDmat.shape)#(5729,800)
# featurelda = []
# for i in range(0,len(lowDmat)):
#     tem = list(lowDmat[i])
#     featurelda.append(tem)
#     print(i)
#     print('################type(tem)',type(tem))#list
#     #print('###############tem:',tem)
# lowtest = lda.transform(feature_test)
# print('降维后的测试数据维度：',lowtest.shape)#(5729,800)
# featuretest = []
# for i in range(0,len(lowtest)):
#     tem = list(lowtest[i])
#     featuretest.append(tem)
#     print(i)
#     print('################testtype(tem)',type(tem))#list
#
# feature_train = list(featurelda)
# feature_test = list(featuretest)
# length =len(feature_train[0])
# print('###########len(feature_train[0]):',length)
# print('len(feature_train):',len(feature_train))
# print('%%%%%%%%%feature_train[0]:',feature_train[0])
# import csv
# with open('../modelsave/transformer_lda_featuretest.csv','w',newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     for row in feature_test:
#         writer.writerow(row)

###############################
#activation = 'sigmoid'
activation = 'relu'
dropout = 0.2
epoch = 20
#初始化的参数
params_dict = {
    'kernel_initializer': 'glorot_uniform',#卷积核的初始化器，权重的初始化方式正态分布
    'kernel_regularizer': l2(0.01),#卷积核的正则化
}
####################################
#transformer
num_layers = 4 #num_layers和num_heads乘积能被model_size整除
model_size = 40
num_heads = 8
dff_size = 128
maxlen = 4
vocab_size = 200 #字典长度，矩阵中数字种类数train31，test58
enc_inputs = keras.layers.Input(shape=(maxlen,))
#dec_inputs = keras.layers.Input(shape=(maxlen,))
transformer = Transformer(num_layers=num_layers, model_size=model_size, num_heads=num_heads, dff_size=dff_size,
                          vocab_size=vocab_size+1, maxlen=maxlen)
final_output = transformer(enc_inputs)
final_output = SpatialDropout1D(0.2)(final_output)
final_output = Convolution1D(filters=64,kernel_size=15, padding='same', kernel_initializer='glorot_normal',
                            kernel_regularizer=l2(0.001))(final_output)
final_output = Activation('relu')(final_output)
final_output = GlobalMaxPooling1D()(final_output)
#final_output = Dropout(dropout)(final_output)
#final_output = MaxPooling1D(3)(final_output)
#final_output = Dense(10,'softmax',**params_dict)(final_output)
#final_output = Flatten()
final_output = Dense(9,'softmax',**params_dict)(final_output)
###########

#模型的入口和出口
model = Model(inputs=enc_inputs,outputs=final_output)
model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])#训练时用的优化器，损失函数，准确率评测标准
print(model.summary())#打印模型参数数量等信息
#model = load_model('model.h5')
#plot_model(model,'model_plot.png')

feature_train = [list(i) for i in feature_train]
feature_test = [list(i) for i in feature_test]

for i in range(epoch):
    print(i)
    model.fit(feature_train,y_train,verbose=1,epochs=i+1,initial_epoch=i,batch_size=64,shuffle=True)
#model.save('../modelsave/transformer_lda_10epoch.h5')

a = model.predict(x=feature_test)

print(a)
print(a[0])
print(type(a[0]))
print(a[0][0])
print(type(a[0][0]))
print('###########len(a)',len(a))
print('#############len(a[0])',len(a[0]))
#outfile = open('../modelsave/cnn_lda_transformer.txt','w')
with open('../modelsave/Adrenalgland1_lda_transformer_float_epoch200.txt','w',newline='') as f:
    for i in range(len(a)):
        f.write(str(i))
        f.write(',')
        for j in range(len(a[i])):
            f.write(str(a[i][j]))
            f.write(',')
        f.write('\n')