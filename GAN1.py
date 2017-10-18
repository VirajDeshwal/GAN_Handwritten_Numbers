#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 02:39:32 2017

@author: virajdeshwal
"""

import pickle as pkl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')


#two input one for discriminator and one for generator 

def model_inputs(real_dim, z_dim):
    input_real =tf.placeholder(tf.float32, (None, real_dim), name = 'input_real')
    input_z = tf.placeholder(tf.float32, (None, z_dim), name ='input_z')
    
    
    return input_real, input_z

#Generator 
def generator(z, out_dim, n_units=128, reuse = False, alpha = 0.01):
    with tf.variable_scope('generator', reuse = reuse):
        #Hidden Layer
        h1 = tf.layers.dense(z, n_units , activation = None)
        #leaky Relu
        h1 =tf.maximum(alpha * h1, h1)
        
        #logits and tanh outputs
        logits = tf.layers.dense(h1, out_dim, activation = None )
        out = tf.tanh(logits)
        
        
        return out
    
#Discriminator
def discriminator(x , n_units=128, reuse = False, alpha = 0.01):
    with tf.variable_scope('discriminator', reuse =reuse):
        #Hidden Layer
        h1=tf.layers.dense(x, n_units, activation = None)
        
        #Leaky Relu
        h1 =tf.maximum(alpha*h1, h1)
        
        logits = tf.layers.dense(h1, 1, activation = None)
        out =tf.sigmoid(logits)
        return out, logits
    
    
#HYPERPARAMETERS
#size of the input image to discriminator
        
input_size = 784
#size of latent vector to generator
z_size =100
#sizes of Hidden Layers in generator and discriminator
g_hidden_size =128
d_hidden_size =128

#leak factor for leaky relu
alpha =0.01
#smoothing
smooth = 0.1


#build the network 
tf.reset_default_graph()

#Create our Input Placeholder
input_real, input_z = model_inputs(input_size , z_size)
#build the model 
g_model =generator (input_z, input_size, n_units =g_hidden_size, alpha= alpha)
#g_model is the generator output

d_model_real, d_logits_real = discriminator(input_real, n_units = d_hidden_size, alpha = alpha)
d_model_fake, d_logits_fake = discriminator(g_model, reuse = True, n_units = d_hidden_size, alpha = alpha)

#Calculate the losses 
d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits= d_logits_real,
                                                labels =tf.ones_like(d_logits_real)*(1-smooth)))

d_loss_fake = tf.reduce_mean(
                  tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, 
                                                          labels=tf.zeros_like(d_logits_real)))
d_loss = d_loss_real + d_loss_fake

g_loss = tf.reduce_mean(
             tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                     labels=tf.ones_like(d_logits_fake)))


#optimizers
learning_rate =0.002
#Get the trainable values split into G and D
t_vars = tf.trainable_variables()
g_vars =[var for var in t_vars if var.name.startswith('generator')]
d_vars =[var for var in t_vars if var.name.startswith('discriminator')]

d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list= d_vars)
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list= g_vars)



#training
batch_size = 100
epochs = 100
samples = []
losses = []
# Only save generator variables
saver = tf.train.Saver(var_list=g_vars)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for ii in range(mnist.train.num_examples//batch_size):
            batch = mnist.train.next_batch(batch_size)
            
            # Get images, reshape and rescale to pass to D
            batch_images = batch[0].reshape((batch_size, 784))
            batch_images = batch_images*2 - 1
            
            # Sample random noise for G
            batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size))
            
            # Run optimizers
            _ = sess.run(d_train_opt, feed_dict={input_real: batch_images, input_z: batch_z})
            _ = sess.run(g_train_opt, feed_dict={input_z: batch_z})
        
        # At the end of each epoch, get the losses and print them out
        train_loss_d = sess.run(d_loss, {input_z: batch_z, input_real: batch_images})
        train_loss_g = g_loss.eval({input_z: batch_z})
            
        print("Epoch {}/{}...".format(e+1, epochs),
              "Discriminator Loss: {:.4f}...".format(train_loss_d),
              "Generator Loss: {:.4f}".format(train_loss_g))    
        # Save losses to view after training
        losses.append((train_loss_d, train_loss_g))
        
        # Sample from generator as we're training for viewing afterwards
        sample_z = np.random.uniform(-1, 1, size=(16, z_size))
        gen_samples = sess.run(
                       generator(input_z, input_size, n_units=g_hidden_size, reuse=True, alpha=alpha),
                       feed_dict={input_z: sample_z})
        samples.append(gen_samples)
        saver.save(sess, './checkpoints/generator.ckpt')

# Save training generator samples
with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)
    
    
#training loss
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses")
plt.legend()