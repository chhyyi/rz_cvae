#!/usr/bin/env python
# coding: utf-8

# # About this notebook
# python script version of chhyyi/rz_cvae/importing_gozsoy.ipynb, to run on gcp.

#Changes:
# will be separated into two part : training, image generation.
# import sys, use sys.exit()

import gozsoy_src.model as gmodel
import gozsoy_src.main as gmain
import gozsoy_src.utils as gutils

import tensorflow as tf

import rz_cvae.Dataset as Dataset
from rz_cvae.utils import save_data
from rz_cvae.CVAE_M1 import CVAE_M1, Decoder, Encoder


import pandas as pd
from sys import exit #to control executions.

# load dataset
table_path = "fp_and_cond_input_19to21.csv"
print("file paths and conditional inputs table file loaded: ", table_path)

learning_rate = 0.001
train_size = 0.5 #between 0.0 ~ 1.0
batch_size = 8
save_test_set = True
# S# True: the test set image IDs and other useful information will be stored in a pickle file to further uses (e.g. Image_Generation.ipynb) 

param1 = dict(table_path = table_path, learning_rate = learning_rate, train_size = train_size, batch_size = batch_size, save_test_set = save_test_set)
for k, v in param1.items():
    print("{} : {}".format(k, v))
if input("continue?(y/n)")=='n':
    exit(0)

dataset = Dataset.Dataset(train_size = train_size, 
                          batch_size = batch_size, 
                          input_table = table_path,
                          save_test_set = save_test_set)
label_dim = 26
image_dim = [1024, 1024, 6]
latent_dim = 256

print("label_dim(cond. input dim) : {}, image_dim : {}, latent_dim : {}".format(label_dim, image_dim, latent_dim))
if input("continue?(y/n)")=='n':
    exit(0)

# Model
model = gmodel.Conditional_VAE(latent_dim)
print("model declared")

# Optiizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
print("optimzer initialized")

#dataset export setup
print(dataset.set_img_include_label(2)

# ## set checkpoint

# Checkpoint path
checkpoint_root = "./1208_1_{}_{}".format(latent_dim, train_size) 
checkpoint_name = "model"
save_prefix = os.path.join(checkpoint_root, checkpoint_name)
print("checkpoint root(directory): {}, checkpoint name: {}".format(checkpoint_root, checkpoint_name))
input("press enter to continue...")

# Define the checkpoint
checkpoint = tf.train.Checkpoint(module=model)

import numpy as np
import time
from rz_cvae.utils import train_step

beta = 1e-4 #it was too low... but why?? Still I don't understand KL loss.


train_losses = []
train_recon_errors = []
train_latent_losses = []
loss = []
reconstruct_loss = []
latent_loss = []

step_index = 0
n_batches = int(dataset.train_size / batch_size)
n_epochs = 30

print("Beta: ", beta)
print("Number of epochs: {},  number of batches: {}".format(n_epochs, n_batches))
input("press enter to start train step...")

# Epochs Loop
for epoch in range(n_epochs):
    start_time = time.perf_counter()
    dataset.shuffle() # Shuffling
    dataset.set_img_include_label(2)

    # Train Step Loop
    for step_index, inputs in enumerate(dataset):
        imgs_with_label, orig_img = tf.split(inputs[0], num_or_size_splits=2, axis=3)
        # training loop
        with tf.GradientTape() as tape:
            # forward pass
            z_mu,z_rho,decoded_imgs = model(imgs_with_label,inputs[1])

            # compute loss
            mse,kl = gmain.elbo(z_mu,z_rho,decoded_imgs,orig_img)
            total_loss = mse + beta * kl

        # compute gradients
        gradients = tape.gradient(total_loss,model.variables)

        # update weights
        optimizer.apply_gradients(zip(gradients, model.variables))

        # compute loss
        train_losses.append(total_loss)
        train_recon_errors.append(mse)
        train_latent_losses.append(kl)

        if step_index + 1 == n_batches:
            break

    loss.append(np.mean(train_losses, 0))
    reconstruct_loss.append(np.mean(train_recon_errors, 0))
    latent_loss.append(np.mean(train_latent_losses, 0))

    exec_time = time.perf_counter() - start_time
    print("Execution time: %0.3f \t Epoch %i: loss %0.4f | reconstr loss %0.4f | latent loss %0.4f"
                        % (exec_time, epoch, loss[epoch], reconstruct_loss[epoch], latent_loss[epoch])) 

    if np.isnan([loss[-1], reconstruct_loss[-1], latent_loss[-1]]).any():
        print("loss diverged. stop training")
        break
    
    # Save progress every 5 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint.save(save_prefix + "_" + str(epoch + 1))
        print("Model saved:", save_prefix)

# Save the final model                
checkpoint.save(save_prefix)
print("Model saved:", save_prefix)
