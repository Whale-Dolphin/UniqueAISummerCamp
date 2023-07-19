import numpy as np

# 加载MNIST数据集
def load_data():
    with open('train-images-idx3-ubyte', 'rb') as f:
        magic, num, rows, cols = np.fromfile(f, dtype=np.uint32, count=4)
        train_images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)

    with open('train-labels-idx1-ubyte', 'rb') as f:
        magic, num = np.fromfile(f, dtype=np.uint32, count=2)
        train_labels = np.fromfile(f, dtype=np.uint8)

    with open('t10k-images-idx3-ubyte', 'rb') as f:
        magic, num, rows, cols = np.fromfile(f, dtype=np.uint32, count=4)
        test_images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)

    with open('t10k-labels-idx1-ubyte', 'rb') as f:
        magic, num = np.fromfile(f, dtype=np.uint32, count=2)
        test_labels = np.fromfile(f, dtype=np.uint8)

    return train_images, train_labels, test_images, test_labels

# 对数据进行预处理
def preprocess(train_images, train_labels, test_images, test_labels):
    # 将图像数据转换为浮点数，并归一化到0-1之间
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    # 将图像数据转换为4D张量，以便输入到CNN中
    train_images = np.expand_dims(train_images, axis=3)
    test_images = np.expand_dims(test_images, axis=3)

    # 将标签数据转换为one-hot编码
    num_classes = 10
    train_labels = np.eye(num_classes)[train_labels]
    test_labels = np.eye(num_classes)[test_labels]

    return train_images, train_labels, test_images, test_labels

# 定义CNN模型
def create_model():
    model = {}

    # 第一层卷积层
    model['conv1'] = {}
    model['conv1']['filters'] = 32
    model['conv1']['kernel_size'] = (3, 3)
    model['conv1']['activation'] = 'relu'
    model['conv1']['input_shape'] = (28, 28, 1)

    # 第二层卷积层
    model['conv2'] = {}
    model['conv2']['filters'] = 64
    model['conv2']['kernel_size'] = (3, 3)
    model['conv2']['activation'] = 'relu'

    # 第三层池化层
    model['pool1'] = {}
    model['pool1']['pool_size'] = (2, 2)

    # 第四层卷积层
    model['conv3'] = {}
    model['conv3']['filters'] = 128
    model['conv3']['kernel_size'] = (3, 3)
    model['conv3']['activation'] = 'relu'

    # 第五层池化层
    model['pool2'] = {}
    model['pool2']['pool_size'] = (2, 2)

    # 第六层全连接层
    model['fc1'] = {}
    model['fc1']['units'] = 128
    model['fc1']['activation'] = 'relu'

    # 第七层全连接层
    model['fc2'] = {}
    model['fc2']['units'] = 10
    model['fc2']['activation'] = 'softmax'

    return model

# 定义卷积层
def conv_layer(x, filters, kernel_size, activation):
    # 获取输入数据的形状
    input_shape = x.shape[1:]

    # 初始化卷积核权重和偏置项
    w = np.random.randn(filters, *kernel_size) / np.sqrt(np.prod(kernel_size))
    b = np.zeros((1, 1, 1, filters))

    # 卷积操作
    z = np.zeros((x.shape[0], input_shape[0] - kernel_size[0] + 1, input_shape[1] - kernel_size[1] + 1, filters))
    for i in range(z.shape[3]):
        for j in range(x.shape[0]):
            for k in range(z.shape[1]):
                for l in range(z.shape[2]):
                    z[j, k, l, i] = np.sum(x[j, k:k+kernel_size[0], l:l+kernel_size[1]] * w[i]) + b[0, 0, 0, i]

    # 激活函数
    if activation == 'relu':
        a = np.maximum(z, 0)
    elif activation == 'sigmoid':
        a = 1 / (1 + np.exp(-z))
    else:
        a = z

    return a

# 定义池化层
def pool_layer(x, pool_size):
    # 获取输入数据的形状
    input_shape = x.shape[1:]

    # 池化操作
    z = np.zeros((x.shape[0], input_shape[0] // pool_size[0], input_shape[1] // pool_size[1], x.shape[3]))
    for i in range(z.shape[3]):
        for j in range(x.shape[0]):
            for k in range(z.shape[1]):
                for l in range(z.shape[2]):
                    z[j, k, l, i] = np.max(x[j, k*pool_size[0]:(k+1)*pool_size[0], l*pool_size[1]:(l+1)*pool_size[1], i])

    return z

# 定义全连接层
def fc_layer(x, units, activation):
    # 获取输入数据的形状
    input_shape = x.shape[1:]

    # 初始化权重和偏置项
    w = np.random.randn(np.prod(input_shape), units) / np.sqrt(np.prod(input_shape))
    b = np.zeros((1, units))

    # 展开输入数据
    x = x.reshape(x.shape[0], -1)

    # 全连接操作
    z = np.dot(x, w) + b

    # 激活函数
    if activation == 'relu':
        a = np.maximum(z, 0)
    elif activation == 'sigmoid':
        a = 1 / (1 + np.exp(-z))
    else:
        a = z

    return a

# 定义softmax函数
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# 定义交叉熵损失函数
def cross_entropy_loss(y_pred, y_true):
    return -np.mean(y_true * np.log(y_pred + 1e-8))

# 定义准确率函数
def accuracy(y_pred, y_true):
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))

# 加载数据集
train_images, train_labels, test_images, test_labels = load_data()

# 对数据进行预处理
train_images, train_labels, test_images, test_labels = preprocess(train_images, train_labels, test_images, test_labels)

# 定义CNN模型
model = create_model()

# 训练模型
learning_rate = 0.01
num_epochs = 10
batch_size = 128
num_batches = train_images.shape[0] // batch_size

for epoch in range(num_epochs):
    for batch in range(num_batches):
        # 获取当前批次的数据
        x_batch = train_images[batch*batch_size:(batch+1)*batch_size]
        y_batch = train_labels[batch*batch_size:(batch+1)*batch_size]

        # 前向传播
        conv1 = conv_layer(x_batch, **model['conv1'])
        conv2 = conv_layer(conv1, **model['conv2'])
        pool1 = pool_layer(conv2, **model['pool1'])
        conv3 = conv_layer(pool1, **model['conv3'])
        pool2 = pool_layer(conv3, **model['pool2'])
        fc1 = fc_layer(pool2, **model['fc1'])
        fc2 = fc_layer(fc1, **model['fc2'])
        y_pred = softmax(fc2)

        # 计算损失和准确率
        loss = cross_entropy_loss(y_pred, y_batch)
        acc = accuracy(y_pred, y_batch)

        # 反向传播
        delta = y_pred - y_batch
        delta = fc_layer(delta, units=model['fc2']['units'], activation='linear')
        delta = fc_layer(delta, units=model['fc1']['units'], activation='relu')
        delta = delta.reshape(pool2.shape)
        delta = pool_layer(delta, pool_size=model['pool2']['pool_size'])
        delta = conv_layer(delta, filters=model['conv3']['filters'], kernel_size=model['conv3']['kernel_size'], activation='relu')
        delta = pool_layer(delta, pool_size=model['pool1']['pool_size'])
        delta = conv_layer(delta, filters=model['conv2']['filters'], kernel_size=model['conv2']['kernel_size'], activation='relu')
        delta = conv_layer(delta, filters=model['conv1']['filters'], kernel_size=model['conv1']['kernel_size'], activation='relu')

        # 更新权重和偏置项
        model['fc2']['weights'] -= learning_rate * np.dot(fc1.T, delta) / batch_size
        model['fc2']['biases'] -= learning_rate * np.mean(delta, axis=(0, 1))
        delta = fc_layer(delta, units=model['fc1']['units'], activation='relu')
        model['fc1']['weights'] -= learning_rate * np.dot(pool2.reshape(pool2.shape[0], -1).T, delta) / batch_size
        model['fc1']['biases'] -= learning_rate * np.mean(delta, axis=0)
        delta = delta.reshape(pool2.shape)
        delta = pool_layer(delta, pool_size=model['pool2']['pool_size'])
        delta = conv_layer(delta, filters=model['conv3']['filters'], kernel_size=model['conv3']['kernel_size'], activation='relu')
        delta = pool_layer(delta, pool_size=model['pool1']['pool_size'])
        delta = conv_layer(delta, filters=model['conv2']['filters'], kernel_size=model['conv2']['kernel_size'], activation='relu')
        model['conv2']['weights'] -= learning_rate * np.array([np.dot(conv1[:, i:i+model['conv2']['kernel_size'][0], j:j+model['conv2']['kernel_size'][1]].reshape(conv1.shape[0], -1).T, delta[:, i, j, :]) / batch_size for i in range(delta.shape[1]) for j in range(delta.shape[2])]).transpose((1, 2, 3, 0))
        model['conv2']['biases'] -= learning_rate * np.mean(delta, axis=(0, 1, 2))
        delta = conv_layer(delta, filters=model['conv1']['filters'], kernel_size=model['conv1']['kernel_size'], activation='relu')
        model['conv1']['weights'] -= learning_rate * np.array([np.dot(x_batch[:, i:i+model['conv1']['kernel_size'][0], j:j+model['conv1']['kernel_size'][1]].reshape(x_batch.shape[0], -1).T, delta[:, i, j, :]) / batch_size for i in range(delta.shape[1]) for j in range(delta.shape[2])]).transpose((1, 2, 3, 0))
        model['conv1']['biases'] -= learning_rate * np.mean(delta, axis=(0, 1, 2))

    print('Epoch %d: loss=%.4f, acc=%.4f' % (epoch+1, loss, acc))

# 在测试集上评估模型
conv1 = conv_layer(test_images, **model['conv1'])
conv2 = conv_layer(conv1, **model['conv2'])
pool1 = pool_layer(conv2, **model['pool1'])
conv3 = conv_layer(pool1, **model['conv3'])
pool2 = pool_layer(conv3, **model['pool2'])
fc1 = fc_layer(pool2, **model['fc1'])
fc2 = fc_layer(fc1, **model['fc2'])
y_pred = softmax(fc2)

test_loss = cross_entropy_loss(y_pred, test_labels)
test_acc = accuracy(y_pred, test_labels)

print('Test loss=%.4f, acc=%.4f' % (test_loss, test_acc))
# copilot写的初步看了下有点小问题，我再来改一改QAQ