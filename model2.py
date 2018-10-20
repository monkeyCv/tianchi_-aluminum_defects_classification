# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
import numpy as np
import myDensenet
from data_process import get_class_correspond

def model(classes,height,width,channels):
    regular_weight=0.105
    base_net=myDensenet.DenseNet121(input_shape=(height,width,channels),include_top=False,weights='imagenet')
    x=base_net.output
    x=keras.layers.Flatten()(x)
    x=keras.layers.Dense(200,activation='relu',kernel_regularizer=keras.regularizers.l2(regular_weight))(x)
    x=keras.layers.Dropout(0.5)(x)
    predictions=keras.layers.Dense(classes,activation='softmax',kernel_regularizer=keras.regularizers.l2(regular_weight))(x)
    net=keras.models.Model(inputs=base_net.input,outputs=predictions)
    return net

def focal_loss(gamma=2.,alpha=0.9):
    def focal_loss_fixed_and_uniform_distribution(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        loss_uniform=tf.reduce_mean(tf.log(y_pred)*tf.ones_like(y_pred),axis=0)
        return -alpha*tf.reduce_sum( tf.pow(1. - pt_1, gamma) * tf.log(pt_1),axis=0) \
            -(1-alpha)*loss_uniform
    return focal_loss_fixed_and_uniform_distribution

def train(train_tfrecord,val_tfrecord,height,width,channels,classes,norm_mean,norm_std):
    train_dataset=tf.data.TFRecordDataset(train_tfrecord,num_parallel_reads=4)
    def train_parse_function(example_proto):
        features={'image_raw':tf.FixedLenFeature((),tf.string,default_value=''),
                  'label':tf.FixedLenFeature((),tf.int64,default_value=0)
                }
        parsed_features=tf.parse_single_example(example_proto,features)
        image=tf.decode_raw(parsed_features['image_raw'],tf.uint8)
        image=tf.reshape(image,shape=(height,width,channels))
        image=tf.cast(image,tf.float32)
        image=image/255.0
        image=image-norm_mean
        image=image/norm_std;
        
        p0=tf.random_normal([])
        angle=tf.random_normal(shape=[],mean=0,stddev=30)
        image=tf.cond(tf.greater(p0,0.5),lambda:tf.contrib.image.rotate(image,angle),
              lambda:tf.identity(image))
        p1=tf.random_normal([])

        dx_dy=tf.random_normal(shape=[2,],mean=[0,0],stddev=[30,30])
        image=tf.cond(tf.greater(p1,0.5),lambda:tf.contrib.image.translate(image,dx_dy),
              lambda:tf.identity(image))
        p2=tf.random_normal([])
        image=tf.cond(tf.greater(p2,0.5),lambda:tf.image.random_flip_up_down(image),
                      lambda:tf.identity(image))
        p3=tf.random_normal([])
        image=tf.cond(tf.greater(p3,0.5),lambda:tf.image.random_flip_left_right(image),
                      lambda:tf.identity(image))
        p4=tf.random_normal([])
        image=tf.cond(tf.greater(p4,0.5),lambda:tf.image.random_hue(image,0.1),
                      lambda:tf.identity(image))
        p5=tf.random_normal([])
        image=tf.cond(tf.greater(p5,0.5),lambda:tf.maximum(0.0,tf.minimum(tf.image.random_brightness(image,0.15),1.0)),
              lambda:tf.identity(image))
        p6=tf.random_normal([])
        image=tf.cond(tf.greater(p6,0.5),lambda:tf.image.random_contrast(image,0.8,1.2),
              lambda:tf.identity(image))
        p7=tf.random_normal([])
        image=tf.cond(tf.greater(p7,0.5),lambda:tf.image.random_saturation(image,0.8,1.0),
              lambda:tf.identity(image))
        
        label=parsed_features['label']
        label=tf.one_hot(label,depth=classes)
        return image,label
    train_dataset=train_dataset.map(train_parse_function)
    train_dataset=train_dataset.repeat().shuffle(5000)
    train_dataset=train_dataset.batch(15)
    
    val_dataset=tf.data.TFRecordDataset(val_tfrecord,num_parallel_reads=4)
    def val_parse_function(example_proto):
        features={'image_raw':tf.FixedLenFeature((),tf.string,default_value=''),
                  'label':tf.FixedLenFeature((),tf.int64,default_value=0)
                }
        parsed_features=tf.parse_single_example(example_proto,features)
        image=tf.decode_raw(parsed_features['image_raw'],tf.uint8)
        image=tf.reshape(image,shape=(height,width,channels))
        image=tf.cast(image,tf.float32)
        image=image/255.0
        image=image-norm_mean
        image=image/norm_std;
        label=parsed_features['label']
        label=tf.one_hot(label,depth=classes)
        return image,label
    val_dataset=val_dataset.map(val_parse_function).repeat()
    val_dataset=val_dataset.batch(9)
    
    net=model(classes,height,width,channels)
    
    print("fine-tune for all layers!")
    for layer in net.layers:
        layer.tainable=False
    for layer in net.layers[0:5]:
        layer.tainable=True
    ckpt = tf.train.get_checkpoint_state('../model')
    if not ckpt or   not ckpt.model_checkpoint_path:
        print("No model! Start to train the model!")
    else:
        net.load_weights('../model/mine')
        print("Model is loaded!")
    net.compile(optimizer=keras.optimizers.Adam(),loss=focal_loss(gamma=2.),metrics=['accuracy'])
    def my_lr(epoch):
        base_lr=6e-7
        lr=base_lr*(0.1**(epoch//5))
        return lr  
    
    net.fit(train_dataset,epochs=15,steps_per_epoch=800,validation_data=val_dataset,
            callbacks=[keras.callbacks.LearningRateScheduler(my_lr),
                       keras.callbacks.ModelCheckpoint('../model/mine', monitor='val_loss',
                                                       verbose=0, save_best_only=True,
                                                       save_weights_only=False,
                                                       mode='auto',period=1)],validation_steps=23)
    info=net.evaluate(val_dataset,steps=23)

    return info

def test(test_tfrecord,norm_mean,norm_std):
    test_dataset=tf.data.TFRecordDataset(test_tfrecord,num_parallel_reads=4)
    def test_parse_function(example_proto):
        features={'image_raw':tf.FixedLenFeature((),tf.string,default_value=''),
                  'label':tf.FixedLenFeature((),tf.int64,default_value=0)
                }
        parsed_features=tf.parse_single_example(example_proto,features)
        image=tf.decode_raw(parsed_features['image_raw'],tf.uint8)
        image=tf.reshape(image,shape=(height,width,channels))
        image=tf.cast(image,tf.float32)
        image=image/255.0
        image=image-norm_mean
        image=image/norm_std
        label=parsed_features['label']
        label=tf.one_hot(label,depth=classes)
        return image,label
    test_dataset=test_dataset.map(test_parse_function)
    test_dataset=test_dataset.batch(1)
    
    net=model(classes,height,width,channels)
    net.compile(optimizer=keras.optimizers.Adam(),loss='categorical_crossentropy',metrics=['accuracy'])
    ckpt = tf.train.get_checkpoint_state('../model')
    if not ckpt or   not ckpt.model_checkpoint_path:
        print("No model! Start to train the model!")
    else:
        net.load_weights('../model/mine')
        print("Model is loaded!")
    rst=net.predict(test_dataset,steps=1000)
    return rst
     
def parse_result(rst,diction):
    raw_labels=np.argmax(rst,axis=1)
    raw_labels=raw_labels.tolist()
    labels=[diction[item] for item in raw_labels]
    labels=np.array(labels)
    labels=np.reshape(labels,[-1,1])
    np.savetxt("mine.csv", labels, delimiter=',',fmt='%s')
    return labels

if __name__=='__main__':
    train_tfrecord='..\\data\\train.tfrecords'
    val_tfrecord='..\\data\\val.tfrecords'
    test_tfrecord='..\\data\\test2.tfrecords'
    height=240
    width=320
    channels=3
    classes=12
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    info=train(train_tfrecord,val_tfrecord,height,width,channels,classes,mean,std)
    print(info)
    # dic=get_class_correspond()
    # rst=test(test_tfrecord,mean,std)
    # corresponds={0:'defect1',
                 # 1:'defect11',
                 # 2:'defect8',
                 # 3:'defect2',
                 # 4:'defect4',
                 # 5:'defect3',
                 # 6:'norm',
                 # 7:'defect9',
                 # 8:'defect5',
                 # 9:'defect6',
                 # 10:'defect10',
                 # 11:'defect7'}
    # rst2=parse_result(rst,corresponds)
    
    