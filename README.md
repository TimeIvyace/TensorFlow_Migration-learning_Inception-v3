# TensorFlow_Migration-learning_Inception-v3
Using TensorFlow to do migration learning with Inception-v3

inception_model中有训练好的inception-v3模型：tensorflow_inception_graph.pb。

flower_photos是训练集文件。

bottleneck文件夹中是所有图片通过inception-v3的瓶颈层输出（即特征向量），可选择下不下载，若下载，则会网络不会再次计算图片的特征向量，减少训练时间。

migration learning.py是python程序，使用的框架为TensorFlow。

详细代码内容理解可见：http://blog.csdn.net/gaoyueace/article/details/79222225
