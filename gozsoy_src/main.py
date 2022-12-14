import numpy as np
import tensorflow as tf

from .model import Conditional_VAE
from .utils import generate_conditioned_digits

# closed form kl loss computation between variational posterior q(z|x) and unit Gaussian prior p(z) 
def kl_loss(z_mu,z_rho):
    sigma_squared = tf.math.softplus(z_rho) ** 2
    kl_1d = -0.5 * (1 + tf.math.log(sigma_squared) - z_mu ** 2 - sigma_squared)

    # sum over sample dim, average over batch dim
    kl_batch = tf.reduce_mean(tf.reduce_sum(kl_1d,axis=1))

    return kl_batch

def elbo(z_mu,z_rho,decoded_img,original_img):
    # reconstruction loss
    mse = tf.reduce_mean(tf.reduce_sum(tf.square(original_img - decoded_img),axis=1))
    # kl loss
    kl = kl_loss(z_mu,z_rho)

    return mse,kl



def train(latent_dim,beta,epochs,train_ds,dataset_mean,dataset_std):
   
    model = Conditional_VAE(latent_dim)

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

    kl_loss_tracker = tf.keras.metrics.Mean(name='kl_loss')
    mse_loss_tracker = tf.keras.metrics.Mean(name='mse_loss')


    for epoch in range(epochs):

        label_list = None
        z_mu_list = None    

        for _,(imgs,labels) in train_ds.enumerate():
            
            imgs_with_label, orig_img = tf.split(imgs, num_or_size_splits=2, axis=3)
            # training loop
            with tf.GradientTape() as tape:
                # forward pass
                z_mu,z_rho,decoded_imgs = model(imgs_with_label,labels)

                # compute loss
                mse,kl = elbo(z_mu,z_rho,decoded_imgs,orig_img)
                loss = mse + beta * kl
            
            # compute gradients
            gradients = tape.gradient(loss,model.variables)

            # update weights
            optimizer.apply_gradients(zip(gradients, model.variables))

            # update metrics
            kl_loss_tracker.update_state(beta * kl)
            mse_loss_tracker.update_state(mse)

            # save encoded means and labels for latent space visualization
            if label_list is None:
                label_list = labels
            else:
                label_list = np.concatenate((label_list,labels))
                
            if z_mu_list is None:
                z_mu_list = z_mu
            else:
                z_mu_list = np.concatenate((z_mu_list,z_mu),axis=0)


        # generate new samples
        generate_conditioned_digits(model,dataset_mean,dataset_std)


        # display metrics at the end of each epoch.
        epoch_kl,epoch_mse = kl_loss_tracker.result(),mse_loss_tracker.result()
        print(f'epoch: {epoch}, mse: {epoch_mse:.4f}, kl_div: {epoch_kl:.4f}')

        # reset metric states
        kl_loss_tracker.reset_state()
        mse_loss_tracker.reset_state()

    return model,z_mu_list,label_list



if __name__ == '__main__':

    beta = 1e-11
    epochs = 10
    latent_dim = 15

    train_ds,dataset_mean,dataset_std = prepare_data()

    model,z_mu_list,label_list = train(latent_dim,beta,epochs,train_ds,dataset_mean,dataset_std)