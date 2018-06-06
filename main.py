from keras.layers import Conv1D, Dense, Dropout, BatchNormalization, MaxPooling1D, Activation, Flatten
from keras.models import Sequential
from keras.utils import plot_model
from keras.regularizers import l2
import preprocess
from keras.callbacks import TensorBoard
import numpy as np

# 训练参数
batch_size = 128
epochs = 20
num_classes = 10
length = 2048
BatchNorm = True # 是否批量归一化
number = 1000 # 每类样本的数量
normal = True # 是否标准化
rate = [0.7,0.2,0.1] # 测试集验证集划分比例

path = r'data\0HP'
x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess.prepro(d_path=path,length=length,
                                                                  number=number,
                                                                  normal=normal,
                                                                  rate=rate,
                                                                  enc=True, enc_step=28)
# 输入卷积的时候还需要修改一下，增加通道数目
x_train, x_valid, x_test = x_train[:,:,np.newaxis], x_valid[:,:,np.newaxis], x_test[:,:,np.newaxis]
# 输入数据的维度
input_shape =x_train.shape[1:]

print('训练样本维度:', x_train.shape)
print(x_train.shape[0], '训练样本个数')
print('验证样本的维度', x_valid.shape)
print(x_valid.shape[0], '验证样本个数')
print('测试样本的维度', x_test.shape)
print(x_test.shape[0], '测试样本个数')

# 定义卷积层
def wdcnn(filters, kernerl_size, strides, conv_padding, pool_padding,  pool_size, BatchNormal):
    """wdcnn层神经元

    :param filters: 卷积核的数目，整数
    :param kernerl_size: 卷积核的尺寸，整数
    :param strides: 步长，整数
    :param conv_padding: 'same','valid'
    :param pool_padding: 'same','valid'
    :param pool_size: 池化层核尺寸，整数
    :param BatchNormal: 是否Batchnormal，布尔值
    :return: model
    """
    model.add(Conv1D(filters=filters, kernel_size=kernerl_size, strides=strides,
                     padding=conv_padding, kernel_regularizer=l2(1e-4)))
    if BatchNormal:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=pool_size, padding=pool_padding))
    return model

# 实例化序贯模型
model = Sequential()
# 搭建输入层，第一层卷积。因为要指定input_shape，所以单独放出来
model.add(Conv1D(filters=16, kernel_size=64, strides=16, padding='same',kernel_regularizer=l2(1e-4), input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

# 第二层卷积

model = wdcnn(filters=32, kernerl_size=3, strides=1, conv_padding='same',
              pool_padding='valid',  pool_size=2, BatchNormal=BatchNorm)
# 第三层卷积
model = wdcnn(filters=64, kernerl_size=3, strides=1, conv_padding='same',
              pool_padding='valid', pool_size=2, BatchNormal=BatchNorm)
# 第四层卷积
model = wdcnn(filters=64, kernerl_size=3, strides=1, conv_padding='same',
              pool_padding='valid', pool_size=2, BatchNormal=BatchNorm)
# 第五层卷积
model = wdcnn(filters=64, kernerl_size=3, strides=1, conv_padding='valid',
              pool_padding='valid', pool_size=2, BatchNormal=BatchNorm)
# 从卷积到全连接需要展平
model.add(Flatten())

# 添加全连接层
model.add(Dense(units=100, activation='relu', kernel_regularizer=l2(1e-4)))
# 增加输出层
model.add(Dense(units=num_classes, activation='softmax', kernel_regularizer=l2(1e-4)))


# 编译模型 评价函数和损失函数相似，不过评价函数的结果不会用于训练过程中
model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# TensorBoard调用查看一下训练情况
tb_cb = TensorBoard(log_dir='logs')

# 开始模型训练
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_valid, y_valid), shuffle=True,
          callbacks=[tb_cb])

# 评估模型
score = model.evaluate(x=x_test, y=y_test, verbose=0)
print("测试集上的损失：", score[0])
print("测试集上的损失:",score[1])
plot_model(model=model, to_file='wdcnn.png', show_shapes=True)