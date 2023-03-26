# 导入必要的库
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.constraints import Constraint

# 读取mdata.txt中的数据
data = np.loadtxt('mdata.txt')

# 分离输入和输出
X = data[:, :2] # 前两列为输入
y = data[:, 2] # 最后一列为输出
print(X)

class ClipConstraint(Constraint):
    def __init__(self, clip_value):
        self.clip_value = clip_value

    def __call__(self, weights):
        return tf.clip_by_value(weights, -self.clip_value, self.clip_value)

    def get_config(self):
        return {'clip_value': self.clip_value}

# 加载或构建模型
try:
    model = keras.models.load_model('mdata.h5') # 尝试加载已有的模型
except:
    model = keras.Sequential()
    model.add(keras.layers.Dense(16, activation='relu', input_shape=(2,), kernel_constraint=ClipConstraint(0.45)))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 用读取的数据进行五千次训练
model.fit(X, y, epochs=5000)

# 保存模型
model.save('mdata.h5')