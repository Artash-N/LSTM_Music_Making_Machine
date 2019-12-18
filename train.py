# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:14:03 2019

@author: Artash Nath
"""

#Libraries Used In Program

import datetime
import numpy as  np
from mido import MidiFile
import pygame.midi
import time
import matplotlib.pyplot as plt
import os
import random
import tensorflow as tf
import keras
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, MultiRNNCell

###############################################################################
############################################################################### 

tf.reset_default_graph()  # Resetting the TF graph variables

#Lists to store midi note sequences of size 24, 32, 48, and 64 notes
trainseq24 = []
trainseq32 = []
trainseq48 = []
trainseq64 = []

num_tracks = []

###############################################################################
############################################################################### 

#Function to convert a midi note to a frequency
def freq(note):
    a = 440
    return (a / 32) * (2 ** ((note - 9) / 12))

#Function to convert a list of midi notes to a list of frequency
def freqlist(notes):
    a = 440 
    temp = []
    for i in notes:
        temp.append((a / 32) * (2 ** ((i - 9) / 12)))
    return temp

###############################################################################
###############################################################################    
print()
print()
print('Importing Midi Training Files...')
print()
print('Please Allow Up to 10 minutes for files to load...')
print()
print('                    ***                          ')
print()



finaltracks = []
directory = r'D:\AI Databases\midi2' # Directory to the MIDI database

###############################################################################
###############################################################################  

#Looping through every MIDI file and extracting all it's major tracks
for filename in (os.listdir(directory)):  
    
    try:
        p = os.path.join(directory, filename)
      
        mid = MidiFile(p)


        tracks = mid.tracks
    
        if len(tracks) < 3:  #Only use the MIDI file if it has more than 2 tracks
            continue
    
        trlist = []        #2 temprary lists to store the tracks
        finaltr = []

        c = 0
        for track in tracks:
            if c != 0 :
                trlist.append(list(track))
        
            c +=1
    
        for track in trlist:
            if len(track)<5:
                continue
        
            temp = []
            for note in track:
                if (note.type == 'note_on'):
                    temp.append(note)
            
            finaltr.append(temp)
    
        temp = []
        for i in finaltr:
            temp.append(len(i))
        
        avg = sum(temp)/len(temp)
    
        for i in finaltr:
        
            if len(i) > avg:
            
                finaltracks.append(i)
                
    except:
        pass
###############################################################################
###############################################################################  
        
print('Finished Loading Files...')        
print('Commencing Training...')
print()
print()
print()
print()
print()
print('-----------------------------------------------------------')
print()
print()
print()
print()
print()

###############################################################################
###############################################################################    

#Extracting the exact note value from all the tracks

for i in finaltracks:
    temp = []
    for no in i:
        temp.append(no.note)
        
    num_tracks.append(temp)
        


#Only keeping a track for training if it's longer than 50 notes
temp = []
for i in num_tracks:
    if len(i)>50:
        temp.append(i)
    
num_tracks = temp


#Dividing the list of tracks into seperate list, each containing a note sequence
# each with a lenght of either 24, 32, 48, or 64 notes. This helps the RNN
# To generalize for different input tune lenghts

for item in num_tracks:
    sk = random.choice([24,32,48,64])
    
    fits = int(len(item)/sk)
    
    for i in range(fits):
        bit = item[:sk]
        
        bit = freqlist(bit)
        
        if sk==24:
            trainseq24.append(bit)
        if sk==32:
            trainseq32.append(bit)
        if sk==48:
            trainseq48.append(bit)
        if sk==64:
            trainseq64.append(bit)
            
    item = item[sk:]
        
        
    
    
    
    
#A function that generates data of a preffered batch size for durng training
    
# Divides the dat into the preffered batch size
        
def gendata(batchsize): 
    sk = random.choice([24,32,48,64])
    a = []
    b = []
    
    if sk ==24:
        for i in range(batchsize):
            d = (trainseq24[random.randint(0,(len(trainseq24)-1))])
            d1 = d[:-1]
            d2 = d[1:]
            a.append(d1)
            b.append(d2)
    if sk ==32:
        for i in range(batchsize):
            d = (trainseq32[random.randint(0,(len(trainseq32)-1))])
            d1 = d[:-1]
            d2 = d[1:]
            a.append(d1)
            b.append(d2)
    if sk ==48:
        for i in range(batchsize):
            d = (trainseq48[random.randint(0,(len(trainseq48)-1))])
            d1 = d[:-1]
            d2 = d[1:]
            a.append(d1)
            b.append(d2)
    if sk ==64:
        for i in range(batchsize):
            d = (trainseq64[random.randint(0,(len(trainseq64)-1))])
            d1 = d[:-1]
            d2 = d[1:]
            a.append(d1)
            b.append(d2)
        
    return  np.reshape(a, (-1, len(a[0]), 1)), np.reshape(b, (-1, len(b[0]), 1))
###############################################################################
###############################################################################    
num_time_steps = None
num_inputs = 1

#num_neurons = 1000
num_units = [1024,1024] # Number of LSTM Cells in each layer

num_outputs =1

learning_rate = 0.005
epochs = 100 #Can be anything

batch_size = 256 

###############################################################################
###############################################################################  

X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
Y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])


cells = [BasicLSTMCell(num_units=n) for n in num_units]
stacked_rnn_cell = MultiRNNCell(cells)
cell = tf.contrib.rnn.OutputProjectionWrapper(stacked_rnn_cell, output_size = num_outputs)


outputs, states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)

loss = tf.reduce_mean(tf.square(outputs - Y))

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

###############################################################################
###############################################################################  


# Use predictx() to predict for a single note continuation
# Use predictmult() to predict for any lenght with passing (list, lenght_to_predict)

def predict(x):
     with tf.Session() as sess:

       saver.restore(sess, "./train_ckpt")

       X_new = np.reshape([x], (1, len(x), 1))
       y_pred = np.reshape((sess.run(outputs, feed_dict = {X:X_new})), (len(x),))
       
       return (plt.plot(  list(range(0, len(x))),   x, c = 'red')  ,plt.scatter(  list(range(1, len(x)+1)) ,y_pred), y_pred)
   
    
    

    
    
    
    
def predictmult(x, lenth):
    
    if lenth <1:
        raise ValueError("A lenth <= to 0 is not valid in this definition")
        
    with tf.Session() as sess:
        saver.restore(sess, "./train_ckpt")

        X_new = np.reshape(x, (1, len(x), 1))
        y_pred = np.reshape((sess.run(outputs, feed_dict = {X:X_new})), (len(x),))
         
        preds = list(y_pred)
         
        for i in range(lenth-1):
         
          X_new = np.reshape(preds[i:], (1, len(preds[i:]), 1))
          y_pred = np.reshape((sess.run(outputs, feed_dict = {X:X_new})), (len(preds[i:]),))
           
          preds.append(y_pred[-1])
         
           
    return (  plt.plot(x, c='red'),
                plt.scatter(list(range(1,len(x)+lenth))  ,preds),)    
    
###############################################################################
###############################################################################  
    
#ACTUAL MODEL TRAINING 
    
bz = batch_size   
    

hist = [] #Storage of all MSE in this list
print()
print()
print()
print()
print()
with tf.Session() as sess:
    
    sess.run(init)
    print(datetime.datetime.now().time())
    for i in range(epochs):
        for ii in range(int(3000/bz)):
            xbatch, ybatch = gendata(bz)
        
            sess.run(train, feed_dict = {X:xbatch, Y:ybatch})
            mse = loss.eval(feed_dict = {X:xbatch, Y:ybatch})
            
            if ii% int((int(3000/bz))/5) ==0:
                print(ii/(int((int(3000/bz))/5)),"/5", ' | ',mse)
        
        if i%1 == 0:
            mse = loss.eval(feed_dict = {X:xbatch, Y:ybatch})
            hist.append(mse)
            print(i, "\MSE", mse)
            
            saver.save(sess, "./train_ckpt")
            
    print(datetime.datetime.now().time())
    
print()
print()
print()
print()
print('-------------------------------------------------------')
print()
print()
print()
print()    
    
