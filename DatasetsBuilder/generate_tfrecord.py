import os
import tensorflow as tf 
import numpy as np

# 这些函数封装的真难用
def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

'''
input_file : '../dataset/scan'
output_file : 
'''

def write(input_file, output_file):
    root_dir = input_file
    img_paths = [x.path for x in os.scandir(root_dir+"/color/") if x.name.endswith('.jpg') or x.name.endswith('.png')]
    num_img=len(img_paths)
    file_names = [(root_dir+'/color/{}.jpg'.format(f)).encode() for f in range(num_img)]
    writer = tf.python_io.TFRecordWriter(output_file) #定义writer，传入目标文件路径
    
    for i in range(num_img-15):
        rgb1_filename = ("{}.jpg".format(i)).encode()
        rgb2_filename = ('/color/'+"{}.jpg".format(i+10)).encode()
        depth1_filename = ('/depth/'+"{}.png".format(i)).encode()
        depth2_filename = ('/depth/'+"{}.png".format(i+10)).encode()
        shape1 = (968, 1296, 3)
        shape2 = (968, 1296, 3)
        c1Tw = np.loadtxt(root_dir+'/pose/'+"{}.txt".format(i),dtype=np.float32).ravel()
        c2Tw = np.loadtxt(root_dir+'/pose/'+"{}.txt".format(i+10),dtype=np.float32).ravel()
        K1 = np.loadtxt(root_dir+'/intrinsic/'+"intrinsic_color.txt",dtype=np.float32).ravel()
        K2 = np.loadtxt(root_dir+'/intrinsic/'+"intrinsic_color.txt",dtype=np.float32).ravel()
        tf_example = tf.train.Example(
            features=tf.train.Features(feature={
                'rgb1_filename': bytes_feature(rgb1_filename),
                'rgb2_filename': bytes_feature(rgb2_filename),
                'depth1_filename': bytes_feature(depth1_filename),
                'depth2_filename': bytes_feature(depth2_filename),
                'shape1': tf.train.Feature(int64_list=tf.train.Int64List(value=list(shape1))),
                'shape2': tf.train.Feature(int64_list=tf.train.Int64List(value=list(shape2))),
                'c1Tw': tf.train.Feature(float_list=tf.train.FloatList(value=c1Tw)),
                'c2Tw': tf.train.Feature(float_list=tf.train.FloatList(value=c2Tw)),
                'K1': tf.train.Feature(float_list=tf.train.FloatList(value=K1)),
                'K2': tf.train.Feature(float_list=tf.train.FloatList(value=K2)),
        }))
        #example序列化，并写入文件
        writer.write(tf_example.SerializeToString())
    writer.close()

if __name__ == "__main__":
    write('../dataset/scan', 're1.tfrecords')