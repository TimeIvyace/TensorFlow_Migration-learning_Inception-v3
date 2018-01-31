# -*- coding: utf-8 -*-

import glob  # 返回一个包含有匹配文件/目录的数组
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# inception-v3瓶颈层的节点个数
BOTTLENECT_TENSOR_SIZE = 2048

# 在谷歌提供的inception-v3模型中，瓶颈层结果的张量名称为'pool_3/_reshape:0'
# 可以使用tensor.name来获取张量名称
BOTTLENECT_TENSOR_NAME = 'pool_3/_reshape:0'

# 图像输入张量所对应的名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

# 下载的谷歌inception-v3模型文件目录
MODEL_DIR = '/tensorflow_google/inception_model'

# 下载的训练好的模型文件名
MODEL_FILE = 'tensorflow_inception_graph.pb'

# 将原始图像通过inception-v3模型计算得到的特征向量保存在文件中，下面定义文件存放地址
CACHE_DIR = '/tensorflow_google/bottleneck'

# 图片数据文件夹，子文件为类别
INPUT_DATA = '/tensorflow_google/flower_photos'

# 验证的数据百分比
VALIDATION_PRECENTAGE = 10
# 测试的数据百分比
TEST_PRECENTAGE = 10

# 定义神经网络的参数
LEARNING_RATE = 0.01
STEPS = 4000
BATCH = 100


# 从数据文件夹中读取所有的图片列表并按训练、验证、测试数据分开
# testing_percentage和validation_percentage指定测试和验证数据集的大小
def create_image_lists(testing_percentage, validation_percentage):
    # 得到的图片放到result字典中，key为类别名称，value为类别下的各个图片（也是字典）
    result = {}
    # 获取当前目录下所有的子目录
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    # sub_dirs中第一个目录是当前目录，即flower_photos，不用考虑
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        # 获取当前目录下所有的有效图片文件
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        # 获取当前文件名
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            # 将分离的各部分组成一个路径名,如/flower_photos/roses/*.JPEG
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.'+extension)
            # glob.glob()返回的是所有路径下的符合条件的文件名的列表
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue

        # 通过目录名获取类别的名称（全部小写）
        label_name = dir_name.lower()
        # 初始化当前类别的训练数据集、测试数据集和验证数据集
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name) #获取当前文件名
            # 随机将数据分到训练数据集、测试数据集以及验证数据集
            chance = np.random.randint(100) #随机返回一个整数
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        # 将当前类别的数据放入结果字典
        result[label_name] = {'dir': dir_name, 'training': training_images,
                              'testing': testing_images, 'validation': validation_images}
    return result


# 通过类别名称、所属数据集和图片编号获取一张图片的地址
# image_lists为所有图片信息，image_dir给出根目录，label_name为类别名称，index为图片编号，category指定图片是在哪个训练集
def get_image_path(image_lists, image_dir, label_name, index, category):
    # 获取给定类别中所有图片的信息
    label_lists = image_lists[label_name]
    # 根据所属数据集的名称获取集合中的全部图片信息
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    # 获取图片的文件名
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    # 最终的地址为数据根目录的地址加上类别的文件夹加上图片的名称
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


# 通过类别名称、所属数据集和图片编号经过inception-v3处理之后的特征向量文件地址
def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category)+'.txt'


# 使用加载的训练好的网络处理一张图片，得到这个图片的特征向量
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    # 将当前图片作为输入，计算瓶颈张量的值
    # 这个张量的值就是这张图片的新的特征向量
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    # 经过卷积神经网络处理的结果是一个四维数组，需要将这个结果压缩成一个一维数组
    bottleneck_values = np.squeeze(bottleneck_values) #从数组的形状中删除单维条目
    return  bottleneck_values


# 获取一张图片经过inception-v3模型处理之后的特征向量
# 先寻找已经计算并且保存的向量，若找不到则计算然后保存到文件
def get_or_create_bottleneck(sess, image_lists, label_name, index, category,
                             jpeg_data_tensor, bottleneck_tensor):
    # 获取一张图片对应的特征向量文件路径
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path) #若不存在则创建
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)

    # 如果这个特征向量文件不存在，则通过inception-v3计算，并存入文件
    if not os.path.exists(bottleneck_path):
        # 获取原始的图片路径
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        # 获取图片内容，对图片的读取
        image_data = gfile.FastGFile(image_path, 'rb').read()
        # 通过inception-v3计算特征向量
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        # 将计算得到的特征向量存入文件,join()连接字符串
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:  #打开文件并写入
            bottleneck_file.write(bottleneck_string)
    else:
        # 直接从文件中获取图片相应的特征向量
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    # 返回特征向量
    return bottleneck_values


# 随机选取一个batch的图片作为训练数据
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category,
                                  jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        # 随机一个类别和图片的编号加入当前的训练数据
        label_index = random.randrange(n_classes)  # 返回指定递增基数集合中的一个随机数，基数缺省值为1，随机类别号
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, image_index,
                                              category, jpeg_data_tensor, bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)

    return bottlenecks, ground_truths


# 获取全部的测试数据，在最终测试的时候在所有测试数据上计算正确率
def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    # 枚举所有类别和每个类别中的测试图片
    for label_index, label_name in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
            # 通过inception-v3计算图片对应的特征向量，并将其加入最终数据的列表
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, index, category,
                                                  jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def main(_):
    # 读取所有图片
    image_lists = create_image_lists(TEST_PRECENTAGE, VALIDATION_PRECENTAGE)
    # image_lists.keys()为dict_keys(['roses', 'sunflowers', 'daisy', 'dandelion', 'tulips'])
    n_classes = len(image_lists.keys()) # 类别数
    # 读取已经训练好的inception-v3模型，谷歌训练好的模型保存在了GraphDef Protocol Buffer中
    # 里面保存了每一个节点取值的计算方法以及变量的取值
    # 对模型的读取,二进制
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        #  新建GraphDef文件,用于临时载入模型中的图
        graph_def = tf.GraphDef()
        # 加载模型中的图
        graph_def.ParseFromString(f.read())
        # 加载读取的inception模型，并返回数据输出所对应的张量以及计算瓶颈层结果所对应的张量
        # 从图上读取张量，同时把图设为默认图
        # Tensor("import/pool_3/_reshape:0", shape=(1, 2048), dtype=float32)
        # Tensor("import/DecodeJpeg/contents:0", shape=(), dtype=string)
        bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECT_TENSOR_NAME,
                                                                                              JPEG_DATA_TENSOR_NAME])

        # 定义新的神经网络输入，这个输入就是新的图片经过inception模型前向传播达到瓶颈层的节点取值,None为了batch服务
        bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECT_TENSOR_SIZE],
                                          name='BottleneckInputPlaceholder')
        # 定义新的标准答案
        ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')

        # 定义一层全连接层来解决新的图片分类问题
        with tf.name_scope('final_training_ops'):
            weights = tf.Variable(tf.truncated_normal([BOTTLENECT_TENSOR_SIZE, n_classes], stddev=0.001))
            biases = tf.Variable(tf.zeros([n_classes]))
            logits = tf.matmul(bottleneck_input, weights) + biases
            final_tensor = tf.nn.softmax(logits)

        # 定义交叉熵损失函数
        # tf.nn.softmax中dim默认为-1,即tf.nn.softmax会以最后一个维度作为一维向量计算softmax
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

        # 计算正确率
        with tf.name_scope('evaluation'):
            correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
            # 平均错误率，cast将bool值转成float
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.Session() as sess:
            init = tf.initialize_all_variables()
            sess.run(init)

            # 训练过程
            for i in range(STEPS):
                # 每次获取一个batch的训练数据
                train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks\
                    (sess, n_classes, image_lists, BATCH, 'training', jpeg_data_tensor, bottleneck_tensor)
                sess.run(train_step, feed_dict={bottleneck_input: train_bottlenecks,
                                                ground_truth_input: train_ground_truth})

                # 在验证数据上测试正确率
                if i % 100 == 0 or i+1 == STEPS:
                    validation_bottlenecks, validation_ground_truth = \
                        get_random_cached_bottlenecks(sess, n_classes, image_lists, BATCH,
                                                      'validation', jpeg_data_tensor, bottleneck_tensor)
                    validation_accuracy = sess.run(evaluation_step,
                                                   feed_dict={bottleneck_input: validation_bottlenecks,
                                                              ground_truth_input: validation_ground_truth})
                    print('Step %d :Validation accuracy on random sampled %d examples = %.1f%%' %
                          (i, BATCH, validation_accuracy*100))

            # 在最后的测试数据上测试正确率
            test_bottlenecks, test_ground_truth = get_test_bottlenecks(sess, image_lists, n_classes,
                                                                       jpeg_data_tensor, bottleneck_tensor)
            test_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: test_bottlenecks,
                                                                 ground_truth_input: test_ground_truth})
            print('Final test accuracy = %.1f%%' % (test_accuracy*100))


if __name__ == '__main__':
    tf.app.run()