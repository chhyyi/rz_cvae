"""
It is modified from copy of src/celeba.py from github repo. 'EleMisi/ContionalVAE'
"""

from .utils import save_data #import inside this package
from collections import OrderedDict
import cv2
from tensorflow.keras.utils import Sequence
import math
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf

class Dataset(Sequence):

    def __init__(self, train_size, batch_size, mode = 'train',
            save_test_set = False,
            input_table="fp_and_cond_input.csv", 
            file_cols=['RI_file', 'lbl_img_path', 'CHL_file'],
                img_size=1024,
                img_include_label=0):
        self.input_table_path = input_table
        self.path_cols = file_cols #list column name of file paths in attr_table
        self.path_cols_iter = []
        self.train_img_ids, self.test_img_ids, self.img_paths, self.cond_inputs_col = self.load(train_size)
        self.batch_size = batch_size
        self.mode = mode
        self.img_size=img_size
        self.train_size = len(self.train_img_ids)
        self.img_include_label = self.set_img_include_label(img_include_label) #choose if iterator will return labelled RI images or not.
        if save_test_set:
            self.save_test_set()

    def load(self, train_dim):
        """
        Read dataset information from table csv file, separate into train, test set.
        attributes : it which will be input data/label of latent variables. dataset except images.

            Returns:
                    - train_img_ids [list] : training data set except for the image paths.
                    - test_img_ids [list]
                    - img_paths [list] : same order with train_img_ids, test_img_ids, 
                    - cond_inputs_col [list] : column names of features which will be conditional input
        """

        print("Loading images id and attributes...")

        file_path = self.input_table_path
        df = pd.read_csv(file_path, index_col = 0)

        #split dataframe of paths and attributes (input/label of latent variables)
        paths_df = df[self.path_cols] 
        df.drop(columns=self.path_cols, inplace=True)

        cond_inputs_col = [x for x in df.columns]
        od = OrderedDict(df.to_dict('index'))
        paths_od = paths_df.to_numpy()#OrderedDict(paths_df.to_dict('index'))
        img_ids = OrderedDict()
        for k,v in od.items():
            img_id=[np.float32(x) for x in v.values()]
            img_ids[k] = img_id
        print("img_ids: {} \nAttributes: {} \n".format(len(img_ids), len(cond_inputs_col)))

        #Splitting
        print("Splitting dataset...\n")
        n_train = int(len(img_ids) * train_dim)
        list_img_ids = list(img_ids.items())
        train_img_ids = list_img_ids[:n_train]
        test_img_ids = list_img_ids[n_train:]

        print("Train set dimension: {} \nTest set dimension: {} \n".format(len(train_img_ids), len(test_img_ids)))

        return train_img_ids, test_img_ids, paths_od, cond_inputs_col

    def set_img_include_label(self, inc_label):
        self.img_include_label=inc_label
        if inc_label==2: #both
            self.path_col_iter=([self.path_cols.index(x) for x in ['lbl_img_path', 'CHL_file']] + [self.path_cols.index(x) for x in ['RI_file', 'CHL_file']])
        elif inc_label==1: #true
            self.path_col_iter=[self.path_cols.index(x) for x in ['lbl_img_path', 'CHL_file']]
        elif inc_label==0: #false
            self.path_col_iter=[self.path_cols.index(x) for x in ['RI_file', 'CHL_file']]
        else:
            print("something is wrong with img_include_label variable")
            raise

        return self.path_col_iter
        
    def next_batch(self, idx):
        """
        Returns a batch of images with their ID as numpy arrays.
        """

        batch_img_ids = [x[1] for x in self.train_img_ids[idx * self.batch_size : (idx + 1) * self.batch_size]]
        images_id = [x[0] for x in self.train_img_ids[idx * self.batch_size : (idx + 1) * self.batch_size]]
        batch_imgs = self.get_images(images_id)

        return np.asarray(batch_imgs, dtype='float32'), np.asarray(batch_img_ids, dtype='float32')


    def preprocess_image(self,image_path, img_resize=1024):
        """
        Resizes and normalizes the target image.
        """

        img = cv2.imread(image_path)
        img = cv2.resize(img, (img_resize, img_resize))
        img = np.array(img, dtype='float32')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img /= 255.0 # Normalization to [0.,1.]

        return img


    def get_images(self, imgs_id, with_label=True):
        """
        Returns the list of concatenated images corresponding to the given IDs.
        """
        imgs = []
            
        for i in imgs_id:
            img = []
            for j in self.path_col_iter:
                image_path = self.img_paths[i][j]
                img.append(self.preprocess_image(image_path, img_resize = self.img_size))
            imgs.append(np.concatenate(tuple(img),axis=-1)) #concatenate images by increasing channel

        return imgs


    def save_test_set(self):
        """
        Saves a pickle file with useful information for the test phase:
            - training size
            - test images IDs
            - file path and conditional input table
            - batch size
        """

        try:
            test_data = {
                'train_size' : self.train_size,
                'test_img_ids' : self.test_img_ids,
                'train_img_ids' :self.train_img_ids,
                'attributes' : self.cond_inputs_col,
                'batch_size' : self.batch_size,
                'img_paths' : self.img_paths
            }

            file_path = "./test_data"
            save_data(file_path, test_data)
        except:
            raise
        print("Test data successfully saved.")


    def shuffle(self):
        """
        Shuffles self.train_img_ids, self.img_paths[:self.train_size]
        """
        train_img_ids_shuffled=self.train_img_ids
        img_paths_shuffled=self.img_paths

        shuffle_index = tf.random.shuffle(range(self.train_size), seed=None, name=None).numpy()
        for i1, idx0 in enumerate(shuffle_index):
            train_img_ids_shuffled[i1]=self.train_img_ids[idx0]
            img_paths_shuffled[i1]=self.img_paths[idx0]
        print("IDs shuffled.")

        self.train_img_ids=train_img_ids_shuffled
        self.img_paths=img_paths_shuffled


    def __len__(self):
        return int(math.ceil(self.train_size / float(self.batch_size)))


    def __getitem__(self, index):
        return self.next_batch(index)
