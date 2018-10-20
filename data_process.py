# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 11:27:58 2018

@author: Yuxi1989
"""

import os,shutil
import numpy as np
import tensorflow as tf
import random
from PIL import Image
from PIL import ImageFile
import re

ImageFile.LOAD_TRUNCATED_IMAGES = True

def move_images():
    '''
    merge all images in the '其他' into one folder
    '''
    cwd=os.path.join('..','data','train','其他')
    for i in os.listdir(cwd):
        local_dir=os.path.join(cwd,i)
        if os.path.isdir(local_dir):
            for img in os.listdir(local_dir):
                img_path=os.path.join(local_dir,img)
                shutil.move(img_path,os.path.join(cwd,img))
    folders=[folder for folder in os.listdir(cwd) if os.path.isdir(os.path.join(cwd,folder))]
    for folder in folders:
        os.rmdir(folder)
        
def get_class_correspond():
    '''
    the relationship between the label name and the integer number
    '''
    cwd=os.path.join('..','data','train')
    categorys=[]
    corresponds={}
    for category in os.listdir(cwd):
        path=os.path.join(cwd,category)
        if os.path.isdir(path):
            categorys.append(category)
    for idx,category in enumerate(categorys):
        corresponds[category]=idx
    return corresponds    

def make_tfrecords(corresponds,nrows,ncols):
    '''
    make tfrecord
    '''
#    train_writer=tf.python_io.TFRecordWriter('..\\data\\train.tfrecords')
#    val_writer=tf.python_io.TFRecordWriter('..\\data\\val.tfrecords')
    test_writer=tf.python_io.TFRecordWriter('..\\data\\test2.tfrecords')
    
#    train_img_paths=[]
#    train_img_labels=[]
#    cwd=os.path.join('..','data','train')
#    categorys=os.listdir(cwd)
#    for category in categorys:
#        if os.path.isdir(os.path.join(cwd,category)):
#            loop_num=1000//len(os.listdir(os.path.join(cwd,category)))
#            for _ in range(loop_num):
#                for img in os.listdir(os.path.join(cwd,category)):
#                    if img.split('.')[-1]=='jpg':
#                        train_img_paths.append(os.path.join(cwd,category,img))
#                        train_img_labels.append(corresponds[category])
#    randnum=random.randint(0,len(train_img_paths))
#    random.seed(randnum)
#    random.shuffle(train_img_paths)
#    random.seed(randnum)
#    random.shuffle(train_img_labels)
#    for i in range(len(train_img_labels)):
#        img=Image.open(train_img_paths[i]).convert('RGB')
#        img=img.resize((ncols,nrows))
#        img_raw=img.tobytes()
#        label=train_img_labels[i]
#        example=tf.train.Example(features=tf.train.Features(feature={
#                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
#                'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}))
#        train_writer.write(example.SerializeToString())
#    train_writer.close()
#    
#    cwd=os.path.join('..','data','val')
#    others=['变形','驳口','打白点','打磨印','返底','划伤','火山口','铝屑',
#            '喷涂碰伤','碰凹','气泡','拖烂','纹粗','油印','油渣','杂色','粘接']
#    for path in os.listdir(cwd):
#        if path.split('.')[-1]=='jpg':
#            img=Image.open(os.path.join(cwd,path))
#            img=img.resize((ncols,nrows))
#            img_raw=img.tobytes()
#            pattern=re.compile(r'[^\u4e00-\u9fa5]')
#            zh=pattern.split(path)
#            if zh[0] in others:
#                label=1
#            else:
#                label=corresponds[zh[0]]
#            example=tf.train.Example(features=tf.train.Features(feature={
#                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
#                'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}))
#            val_writer.write(example.SerializeToString())
#    val_writer.close()
    
    cwd=os.path.join('..','data','test2')
    for path in sorted(os.listdir(cwd),key=lambda i:int(i.split('.')[0])):
        if path.split('.')[-1]=='jpg':
            img=Image.open(os.path.join(cwd,path))
            img=img.resize((ncols,nrows))
            img_raw=img.tobytes()
            label=1
            example=tf.train.Example(features=tf.train.Features(feature={
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))}))
            test_writer.write(example.SerializeToString())
    test_writer.close()
            
def split_train_val():
    '''
    get validation set from the training set
    '''
    cwd=os.path.join('..','data')
    train_path=os.path.join(cwd,'train')
    val_path=os.path.join(cwd,'val')
    if not os.path.exists(val_path):
        os.mkdir(os.path.join(cwd,'val'))
    for category in os.listdir(train_path):
        local_path=os.path.join(train_path,category)
        img_path=os.listdir(local_path)
        cnt=len(img_path)
        img_path=np.array(img_path)
        val_img_path=np.random.choice(img_path,max(1,int(cnt*0.1)),replace=False)
        val_img_path=val_img_path.tolist()
        for img in val_img_path:
            src_path=os.path.join(local_path,img)
            dst_path=os.path.join(val_path,img)
            shutil.move(src_path,dst_path)
    
if __name__=='__main__':
    #move_images()
    corresponds=get_class_correspond()
    #split_train_val()
    make_tfrecords(corresponds,240,320)
    
