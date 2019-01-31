#!/usr/bin/env python
# coding: utf-8
# This script is generated using jupyter lab
# In[1]:


from datetime import datetime
startTime = datetime.now()
import numpy as np
import pandas as pd
import os
import cv2
import math
import glob
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import OrderedDict
os.chdir("path\\to\\models\\research\slim")
from datasets import imagenet
from nets import inception
from preprocessing import inception_preprocessing
from tensorflow.contrib import slim
from datasets import dataset_utils

try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib


# In[2]:


# This is needed to display the images.
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#Function to capture 1 frame for every 2 second from video

def FrameCapture(videoFile,imagesFolder):
    image_file_name = videoFile.rsplit("\\",1)[-1].rsplit(".",1)[-2]
    cap = cv2.VideoCapture(videoFile)
    frameRate = cap.get(cv2.CAP_PROP_FPS) * 2 #frame rate
    count = 1
    while(cap.isOpened()):
        frameId = cap.get(cv2.CAP_PROP_POS_FRAMES) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            output_path = os.path.join(imagesFolder, image_file_name)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            #filename = imagesFolder + "\\" + image_file_name + "_image_" +  str(int(count)) + ".jpg"
            filename = output_path +  "\\" + image_file_name + "_image_" +  str(int(count)) + ".jpg"
            cv2.imwrite(filename, frame)
            count += 1
            
    fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frameCount/fps
        
    print("Done!")
    print('FPS = ' + str(fps))
    print('Total number of frames in video = ' + str(frameCount))
    print('Duration (S) = ' + str(duration))
    print("Total frames captured = ",count)
    cap.release()


# Video duration is 30.04 sec
# There are 25 frames per second in the video
# Therefore, Total frames in video are 25*30.04 = 751
# And we are extracting every 50th frame, that is 1 frame for every 2 second
# Hence we get 17 frames


# In[3]:


# function call to capture frames
videoFile = 'path\\to\\video\\file\\kitchen_lg.mp4'
imagesFolder = 'path\\to\\store\\images'
FrameCapture(videoFile,imagesFolder)


# In[7]:


#download inception model as checkpoint file

url = "http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz"
checkpoints_dir = '/checkpoints_v4'

if not tf.gfile.Exists(checkpoints_dir):
    tf.gfile.MakeDirs(checkpoints_dir)

dataset_utils.download_and_uncompress_tarball(url, checkpoints_dir)

image_size = inception.inception_v4.default_image_size
# The default image size used to train this network is 224x224.


# In[4]:


def write_to_csv(file_path, csv_dict_obj, append_mode=False):
    header = csv_dict_obj.keys()
    csv_dict_obj = [csv_dict_obj]
    with open('{}.csv'.format(file_path), 'a',newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, delimiter=',',fieldnames=header)
        if not append_mode:
            dict_writer.writeheader()
        dict_writer.writerows(csv_dict_obj)


# In[8]:


def GenerateTags(imagesFolder):
    images = glob.glob("%s/*.jpg"%(imagesFolder)) 
    video_file_name = videoFile.rsplit("\\",1)[-1].rsplit(".",1)[-2]
    append_mode = False
    for image in images:
        url = '' 
        tagx = '' 
        probx = ''
        all_tags = []
        file_cnt = 0
    
        with tf.Graph().as_default():
                print("Processing image :",image)
                image_file_name = image.rsplit('\\', 1)[-1]
                image_string = urllib.urlopen("file:///" + image).read()
                image = tf.image.decode_jpeg(image_string, channels=3)
                processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
                processed_images  = tf.expand_dims(processed_image, 0)

                # Create the model, use the default arg scope to configure the batch norm parameters.
                with slim.arg_scope(inception.inception_v4_arg_scope()):
                    logits, _ = inception.inception_v4(processed_images, num_classes=1001, is_training=False,reuse=tf.AUTO_REUSE)
                probabilities = tf.nn.softmax(logits)

                init_fn = slim.assign_from_checkpoint_fn(
                                    os.path.join(checkpoints_dir, 'inception_v4.ckpt'),
                                    slim.get_model_variables('InceptionV4'))

                with tf.Session() as sess:
                    init_fn(sess)
                    np_image, probabilities = sess.run([image, probabilities])
                    probabilities = probabilities[0, 0:]
                    sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]

    #             plt.figure()
    #             plt.imshow(np_image.astype(np.uint8))
    #             plt.axis('off')
    #             plt.show()

                names = imagenet.create_readable_names_for_imagenet_labels()

                for i in range(10):
                    index = sorted_inds[i]
                    tags = names[index].split(',')
                    #print "debug", tags, type(tags)
                    store_tags = []
                    for tagx in tags:
                        tagx = tagx.strip() # remove white space
                        probx = probabilities[index]
                        if (tagx not in store_tags): # lowerbound 5% chance
                            values = OrderedDict()
                            values['tag_captured'] = tagx
                            values['probability'] = probx
                            values['image_file_name'] = image_file_name

                            store_tags.append(tagx)
                            all_tags.append(values)
                print("number of tags captured for " , image_file_name, len(all_tags))
                if len(all_tags) > 0 :
                    file_cnt = file_cnt + 1
                for val in all_tags :
                    if val['image_file_name']:
                        write_to_csv(video_file_name, val, append_mode)
                        append_mode = True
    r = glob.glob('*.csv')
    r = "".join(r)
    output_file = str(os.getcwd()) + "\\" + r
    df = pd.read_csv(output_file)
    df.sort_values(['image_file_name','probability'],ascending= [True , False],inplace=True)
    df.to_csv(os.path.join(imagesFolder,r), index=False)
    print(df.shape)


# In[9]:


imagesFolder = 'path\\to\\folder\\where\\images_are_stored\\images\\kitchen_lg'
GenerateTags(imagesFolder)
print("***** Images processed successfully and tags written to file *****")


# In[11]:


print("Script execution time : " , datetime.now() - startTime)

