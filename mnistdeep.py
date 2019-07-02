from tensorflow.examples.tutorials.mnist import input_data
import json
import tensorflow as tf

#初始化权重函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1) #截断的正态分布，标准差stddev
    return tf.Variable(initial)

#初始化偏置项
def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

#自定义卷积函数
def conv2d(x,W):
    # stride的四个参数：[batch, height, width, channels], [batch_size, image_rows, image_cols, number_of_colors]
    # height, width就是图像的高度和宽度，batch和channels在卷积层中通常设为1
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

#定义一个2*2的最大池化层
def max_pool_2x2(x):
    """
    max_pool(x,ksize,strides,padding)参数含义
        x:input
        ksize:filter，滤波器大小2*2
        strides:步长，2*2，表示filter窗口每次水平移动2格，每次垂直移动2格
        padding:填充方式，补零
    conv2d(x,W,strides=[1,1,1,1],padding='SAME')参数含义与上述类似
        x:input
        W:filter，滤波器大小
        strides:步长，1*1，表示filter窗口每次水平移动1格，每次垂直移动1格
        padding:填充方式，补零('SAME')
    """
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

if __name__ == '__main__':
    #设置占位符，尺寸为样本输入和输出的尺寸
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y_')
    
    # 将输入的x转成一个4D向量，第2、3维对应图片的宽高，最后一维代表图片的颜色通道数
    # 输入的图像为灰度图，所以通道数为1，如果是RGB图，通道数为3
    # tf.reshape(x,[-1,28,28,1])的意思是将x自动转换成28*28*1的数组
    # -1的意思是代表不知道x的shape，它会按照后面的设置进行转换
    x_image = tf.reshape(x,[-1,28,28,1])
    
    #设置第一个卷积层和池化层
    # patch为5*5，in_size为1，即图像的厚度，如果是彩色，则为3，32是out_size，输出的大小-》32个卷积和（滤波器）
    W_conv1 = weight_variable([5, 5, 1, 32]) # 初始权重值
    b_conv1 = bias_variable([32]) # 初始偏置项
    
    # ReLU操作，输出大小为28*28*32
    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1) # 卷积并激活
    # Pooling操作，输出大小为14*14*32
    h_pool1 = max_pool_2x2(h_conv1) # 池化
    
    # 设置第二个卷积层和池化层
    # patch为5*5，in_size为32，即图像的厚度，64是out_size，输出的大小
    W_conv2 = weight_variable([5, 5, 32, 64]) # 初始权重值
    b_conv2 = bias_variable([64]) # 初始偏置项
    
    # ReLU操作，输出大小为14*14*64
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # 将第一层卷积池化后的结果作为第二层卷积的输入
    # Pooling操作，输出大小为7*7*64
    h_pool2 = max_pool_2x2(h_conv2) # 池化
    
    #设置全连接层
    W_fc1 = weight_variable([7 * 7 * 64, 1024]) # 设置全连接层的权重
    b_fc1 = bias_variable([1024]) # 设置全连接层的偏置项
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # 将第二层卷积池化后的结果，转成一个7*7*64的数组
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # 通过全连接之后并激活
    
    # 防止过拟合
    keep_prob = tf.placeholder("float", name='keep_prob')
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    # 输出层
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y_conv') # 构建回归模型
    
    # 使用Tensorflow自带的交叉熵函数
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv)) # 定义交叉熵为损失函数
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # 配置Adam优化器，学习速率为1e-4
    
    # 建立正确率计算表达式
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) # 检测预测和真实标签的匹配程度，返回的是一组Boolean列表，如：[True, False, True, True]
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) # 计算accuracy，通过reduce_mean()函数将这组布尔列表转成[1,0,1,1]，再求均值
    
    # 创建Saver对象，用来保存训练好的模型
    saver = tf.train.Saver() 
    # 下载并读入MNIST数据
    mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) # 初始化全局变量
        for i in range(20000):
            batch = mnist.train.next_batch(50)  # 每次取出50个数进行训练
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))

                #with open('./1.json', 'a') as f:
                    #a = {'num':i,'train_accuracy':float('%.2f'%train_accuracy)}
                    #json.dump(a, f, indent=2)
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        saver.save(sess, './model/model.ckpt')  # 保存模型参数
    
        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

