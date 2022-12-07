"""
Conditional model version2: ConCAVE_M1() class
changes from ConvolutionalVAE.py:
    - remove every MaxPool2D, Upsampling layers and increase kernel strides & size instead.
    - add dropout layer... but where...?
"""

import numpy as np 
import tensorflow as tf
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPool2D, UpSampling2D, Dropout


#########################
#        ENCODER        #
#########################

class Encoder(tf.keras.Model):

    def __init__(self, latent_dim):
    
        super(Encoder, self).__init__()

        self.enc_block_1 = Conv2D( 
                            filters=32, 
                            kernel_size=5,
                            kernel_regularizer='l1',
                            strides=(4, 4), 
                            padding = 'same',
                            kernel_initializer=he_normal())

        self.enc_block_2 = Conv2D( 
                      filters=64, 
                      kernel_size=5, 
                            kernel_regularizer='l1',
                      strides=(4, 4), 
                      padding = 'same',
                      kernel_initializer=he_normal())

        self.enc_block_3 = Conv2D( 
                      filters=128, 
                      kernel_size=5, 
                            kernel_regularizer='l1',
                      strides=(4, 4), 
                      padding = 'same',
                      kernel_initializer=he_normal())

        self.enc_block_4 = Conv2D( 
                      filters=256, 
                      kernel_size=5, 
                            kernel_regularizer='l1',
                      strides=(4, 4), 
                      padding = 'same',
                      kernel_initializer=he_normal())

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(latent_dim + latent_dim, kernel_regularizer='l1')


    def call(self, img_input, lbl_input, latent_dim, is_train):
        dropout_dense = 0.5
        dropout_conv2d = 0.25

         # Encoder block 1
        x = self.enc_block_1(img_input)
        x = BatchNormalization(trainable = is_train)(x)
        x = tf.nn.leaky_relu(x)
        x = Dropout(dropout_conv2d)(x)
        # Encoder block 2
        x = self.enc_block_2(x)
        x = BatchNormalization(trainable = is_train)(x)
        x = tf.nn.leaky_relu(x)
        x = Dropout(dropout_conv2d)(x)
        # Encoder block 3
        x = self.enc_block_3(x)
        x = BatchNormalization(trainable = is_train)(x)
        x = tf.nn.leaky_relu(x)
        x = Dropout(dropout_conv2d)(x)
        # Encoder block 4
        x = self.enc_block_4(x)
        x = BatchNormalization(trainable = is_train)(x)
        x = tf.nn.leaky_relu(x)   
        
        x = Dropout(dropout_conv2d)(x)
        x = self.flatten(x)
        
        x_lbl_concat = tf.concat([x,lbl_input],1)
        x = self.dense(x_lbl_concat)
        return x



#########################
#        DECODER        #
#########################

class Decoder(tf.keras.Model):
    

    def __init__(self, batch_size = 4):

        super(Decoder, self).__init__()

        self.batch_size = batch_size
        self.dense = tf.keras.layers.Dense(4*4*(128+128), kernel_regularizer='l1') # it should match final output of image size of Encoder squared by latent dim.
                                          
        self.reshape = tf.keras.layers.Reshape(target_shape=(4, 4, (128+128)))

        self.dec_block_1 = Conv2DTranspose(
                filters=256,
                kernel_size=5,
                   kernel_regularizer='l1',
                strides=(4, 4),
                padding='same',
                kernel_initializer=he_normal())

        self.dec_block_2 = Conv2DTranspose(
                filters=128,
                kernel_size=5,
                kernel_regularizer='l1',
                strides=(4, 4),
                padding='same',
                kernel_initializer=he_normal())

        self.dec_block_3 = Conv2DTranspose(
                filters=64,
                kernel_size=5,
                kernel_regularizer='l1',
                strides=(4, 4),
                padding='same',
                kernel_initializer=he_normal())

        self.dec_block_4 = Conv2DTranspose(
                filters=32,
                kernel_size=5,
                kernel_regularizer='l1',
                strides=(4, 4),
                padding='same',
                kernel_initializer=he_normal())

        self.dec_block_5 = Conv2DTranspose(
                filters=6, 
                kernel_size=3,
                kernel_regularizer='l1',
                strides=(1, 1), 
                padding='same',
                kernel_initializer=he_normal())

    def __call__(self, z_cond, is_train):
        dropout_dense = 0.5
        dropout_conv2d = 0.25

        # Reshape input
        x = self.dense(z_cond)
        x = tf.nn.leaky_relu(x)
        x = self.reshape(x)
        x = Dropout(dropout_conv2d)(x)

        # Decoder block 1
        x = self.dec_block_1(x)
        x = BatchNormalization(trainable = is_train)(x)
        x = tf.nn.leaky_relu(x)
        x = Dropout(dropout_conv2d)(x)
        
        # Decoder block 2
        x = self.dec_block_2(x)
        x = BatchNormalization(trainable = is_train)(x)
        x = tf.nn.leaky_relu(x)
        x = Dropout(dropout_conv2d)(x)

        # Decoder block 3
        x = self.dec_block_3(x)
        x = BatchNormalization(trainable = is_train)(x)
        x = tf.nn.leaky_relu(x)
        x = Dropout(dropout_conv2d)(x)

        # Decoder block 4
        x = self.dec_block_4(x)
        x = BatchNormalization(trainable = is_train)(x)
        x = tf.nn.leaky_relu(x)

        return self.dec_block_5(x)



#########################
#       Conv-CVAE       #
#########################

class CVAE_M1(tf.keras.Model) :

    def __init__(self, 
        encoder,
        decoder,
        label_dim,
        latent_dim,
        batch_size = 32,
        beta = 1,
        image_dim = [1024, 1024, 6]):

        super(CVAE_M1, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.label_dim = label_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.beta = beta = 1
        self.image_dim = image_dim              

    def kl_loss(self, z_mu,z_rho): #kl loss from gozsoy's cvae github repo
        sigma_squared = tf.math.softplus(z_rho) ** 2
        kl_1d = -0.5 * (1 + tf.math.log(sigma_squared) - z_mu ** 2 - sigma_squared)

        # sum over sample dim, average over batch dim
        kl_batch = tf.reduce_mean(tf.reduce_sum(kl_1d,axis=1))
        return kl_batch

    def __call__(self, input_image, input_label, is_train):
    
        input_img = tf.keras.layers.InputLayer(input_shape=self.image_dim, dtype = 'float32')(input_image)
        input_label = tf.keras.layers.InputLayer(input_shape=(self.label_dim,), dtype = 'float32')(input_label)
        
        z_mean, z_log_var = tf.split(self.encoder(input_img, input_label, self.latent_dim, is_train), num_or_size_splits=2, axis=1)    
        z_cond = self.reparametrization(z_mean, z_log_var, input_label)
        logits = self.decoder(z_cond, is_train)

        recon_img = tf.nn.sigmoid(logits)
        #temp for debugging
        
        # Loss computation #
        #latent_loss = - 0.5 * self.kl_loss(z_mean, z_log_var) # KL divergence
        latent_loss = - 0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1) # KL divergence
        mse=tf.keras.losses.MeanSquaredError()
        try:
            reconstr_loss = np.prod((1024,1024))*mse(tf.keras.backend.flatten(input_img), tf.keras.backend.flatten(recon_img)) #weigthed mse... but why?
        except:
            print('error while input_img shape: {} and recon_img shape: {}'.format(input_img.shape, recon_img.shape, sep='\n\n'))
            raise
        loss = reconstr_loss + self.beta * latent_loss # weighted ELBO loss
        loss = tf.reduce_mean(loss) 

        return {
                    'recon_img': recon_img,
                    'latent_loss': latent_loss,
                    'reconstr_loss': reconstr_loss,
                    'loss': loss,
                    'z_mean': z_mean,
                    'z_log_var': z_log_var
                }

    def reparametrization(self, z_mean, z_log_var, input_label):
        """ Performs the riparametrization trick"""

        eps = tf.random.normal(shape = (input_label.shape[0], self.latent_dim), mean = 0.0, stddev = 1.0)       
        z = z_mean + tf.math.exp(z_log_var * .5) * eps
        z_cond = tf.concat([z, input_label], axis=1) # (batch_size, label_dim + latent_dim)

        return z_cond


