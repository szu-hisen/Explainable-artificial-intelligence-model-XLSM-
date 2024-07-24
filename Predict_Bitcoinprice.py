import numpy as np
import pandas as pd
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.train import Model
from mindspore.nn.loss.loss import MSELoss
from mindspore.common.initializer import Normal
from mindspore.ops import operations as P
import mindspore.dataset as ds
from mindspore.dataset import GeneratorDataset
from mindspore.dataset.transforms import TypeCast
import mindspore.common.dtype as mstype

# 准备数据
df = pd.read_csv("./data/BTC.csv")
X = df[["Open", "High", "Low", "Volume"]].values  # 特征数据
y = df["Close"].values  # 目标价格数据

# 划分数据集为训练集和测试集
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 创建MindSpore数据集
batch_size = 32

def data_generator(X, y):
    for features, target in zip(X, y):
        yield (features, target)

train_dataset = GeneratorDataset(data_generator(X_train, y_train), column_names=["features", "target"])
test_dataset = GeneratorDataset(data_generator(X_test, y_test), column_names=["features", "target"])

# 对数据进行类型转换
train_dataset = train_dataset.map(operations=TypeCast(mstype.float32))
test_dataset = test_dataset.map(operations=TypeCast(mstype.float32))

# 设置批处理大小并删除余数
train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

# 构建神经网络模型
class BitcoinPricePredictNet(nn.Cell):
    def __init__(self):
        super(BitcoinPricePredictNet, self).__init__()
        self.fc = nn.Dense(4, 1, Normal(0.02), Normal(0.02))

    def construct(self, x):
        x = self.fc(x)
        return x

# 定义损失函数和优化器
net = BitcoinPricePredictNet()
criterion = nn.loss.MSELoss()
optimizer = nn.Momentum(params=net.trainable_params(), learning_rate=0.001, momentum=0.9)
model = Model(net, optimizer)

model.train(epochs=50, train_dataset=X, train_labels=y)
predicted = net(X)