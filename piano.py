# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 17:43:33 2019

@author: Artash Nath
"""

import tensorflow as tf
import numpy as  np
import matplotlib.pyplot as plt
from tensorflow.contrib.rnn import BasicRNNCell, BasicLSTMCell, MultiRNNCell
import math
import sys
import pygame
from pygame.locals import *
from time import sleep
import pygame.midi
pygame.init()
pygame.midi.init()
player = pygame.midi.Output(0)

notes = {'c1':60, 'c#1':61, 'd1':62,'d#1':63, 'e1':64, 'f1':65, 'f#1':66, 'g1':67, 'g#1':68,
         'a1':69, 'a#1':70, 'b1':71,
         
         'c2':72, 'c#2':73, 'd2':74,'d#2':75, 'e2':76, 'f2':77, 'f#2':78, 'g2':79, 'g#2':80,
         'a2':81, 'a#2':82, 'b2':83}

def freq(note):
    
    a = 440 #frequency of A (coomon value is 440Hz)
    temp = ((a / 32) * (2 ** ((note - 9) / 12)))
    return temp


def freqlist(notes):
    
    a = 440 #frequency of A (coomon value is 440Hz)
    temp = []
    for i in notes:
        temp.append((a / 32) * (2 ** ((i - 9) / 12)))
    return temp

def noted(f):
    
    if f ==0:
        return 0 
    d=69+(12*math.log(int(f)/440.0))/(math.log(2))
    return d

def playnoteardu(x,y):
            global n 
            
            if ((0<x<50) and (y<250) ) or ((350>y>250)and(0<x<70)):
                player.note_on(int((notes['c1'])),127)
                sleep(0.27)
                player.note_off(int((notes['c1'])),127)
                n = notes['c1']

                
            if ((93<x<139) and (y<250) ) or ((350>y>250)and(75<x<158)):
                player.note_on(int((notes['d1'])),127)
                sleep(0.27)
                player.note_off(int((notes['d1'])),127)
                n = notes['d1']

            
            if ((181<x<228) and (y<250) ) or ((350>y>250)and(161<x<230)):
                player.note_on(int((notes['e1'])),127)
                sleep(0.27)
                player.note_off(int((notes['e1'])),127)
                n = notes['e1']

                
            if ((233<x<285) and (y<250) ) or ((350>y>250)and(233<x<307)):
                player.note_on(int((notes['f1'])),127)
                sleep(0.27)
                player.note_off(int((notes['f1'])),127)
                n = notes['f1']

                
            if ((328<x<375) and (y<250) ) or ((350>y>250)and(310<x<393)):
                player.note_on(int((notes['g1'])),127)
                sleep(0.27)
                player.note_off(int((notes['g1'])),127)
                n = notes['g1']

            
            if ((415<x<463) and (y<250) ) or ((350>y>250)and(397<x<483)):
                player.note_on(int((notes['a1'])),127)
                sleep(0.27)
                player.note_off(int((notes['a1'])),127)
                n = notes['a1']

                
            if ((505<x<564) and (y<250) ) or ((350>y>250)and(488<x<564)):
                player.note_on(int((notes['b1'])),127)
                sleep(0.27)
                player.note_off(int((notes['b1'])),127)
                n = notes['b1']

                
            if ((568<x<626) and (y<250) ) or ((350>y>250)and(568<x<647)):
                player.note_on(int((notes['c2'])),127)
                sleep(0.27)
                player.note_off(int((notes['c2'])),127)
                n = notes['c2']

                
            if ((670<x<714) and (y<250) ) or ((350>y>250)and(650<x<732)):
                player.note_on(int((notes['d2'])),127)
                sleep(0.27)
                player.note_off(int((notes['d2'])),127)
                n = notes['d2']

                
            if ((756<x<805) and (y<250) ) or ((350>y>250)and(737<x<804)):
                player.note_on(int((notes['e2'])),127)
                sleep(0.27)
                player.note_off(int((notes['e2'])),127)
                n = notes['e2']

            if ((807<x<863) and (y<250) ) or ((350>y>250)and(807<x<880)):
                player.note_on(int((notes['f2'])),127)
                sleep(0.27)
                player.note_off(int((notes['f2'])),127)
                n = notes['f2']

                
            if ((904<x<949) and (y<250) ) or ((350>y>250)and(886<x<968)):
                player.note_on(int((notes['g2'])),127)
                sleep(0.27)
                player.note_off(int((notes['g2'])),127)
                n = notes['g2']

                
            if ((991<x<1038) and (y<250) ) or ((350>y>250)and(972<x<1059)):
                player.note_on(int((notes['a2'])),127)
                sleep(0.27)
                player.note_off(int((notes['a2'])),127)
                n = notes['a2']

                
            if ((1080<x<1149) and (y<250) ) or ((350>y>250)and(1062<x<1149)):
                player.note_on(int((notes['b2'])),127)
                sleep(0.27)
                player.note_off(int((notes['b2'])),127)
                n = notes['b2']
            #############################################
                
            if ((53<x<92) and (y<250)):
                player.note_on(int((notes['c#1'])),127)
                sleep(0.27)
                player.note_off(int((notes['c#1'])),127)
                n = notes['c#1']

                           
            if ((140<x<180) and (y<250)):
                player.note_on(int((notes['d#1'])),127)
                sleep(0.27)
                player.note_off(int((notes['d#1'])),127)
                n = notes['d#1']

                           
            if ((287<x<327) and (y<250)):
                player.note_on(int((notes['f#1'])),127)
                sleep(0.27)
                player.note_off(int((notes['f#1'])),127)
                n = notes['f#1']

                
            if ((376<x<414) and (y<250)):
                player.note_on(int((notes['g#1'])),127)
                sleep(0.27)
                player.note_off(int((notes['g#1'])),127)
                n = notes['g#1']

                           
            if ((466<x<503) and (y<250)):
                player.note_on(int((notes['a#1'])),127)
                sleep(0.27)
                player.note_off(int((notes['a#1'])),127)
                n = notes['a#1']

            if ((629<x<666) and (y<250)):
                player.note_on(int((notes['c#2'])),127)
                sleep(0.27)
                player.note_off(int((notes['c#2'])),127)
                n = notes['c#2']  
      
            if ((716<x<754) and (y<250)):
                player.note_on(int((notes['d#2'])),127)
                sleep(0.27)
                player.note_off(int((notes['d#2'])),127)
                n = notes['d#2'] 
   
                
            if ((865<x<902) and (y<250)):
                player.note_on(int((notes['f#2'])),127)
                sleep(0.27)
                player.note_off(int((notes['f#2'])),127)
                n = notes['f#2']

                
            if ((951<x<989) and (y<250)):
                player.note_on(int((notes['g#2'])),127)
                sleep(0.27)
                player.note_off(int((notes['g#2'])),127)
                n = notes['g#2']    
     
            if ((1041<x<1079) and (y<250)):
                player.note_on(int((notes['a#2'])),127)
                sleep(0.27)
                player.note_off(int((notes['a#2'])),127)
                n = notes['a#2']

                          
            return n
##########################################################################
##########################################################################
            
def playnote(x,y):
            global n 
            
            if ((0<x<50) and (y<250) ) or ((350>y>250)and(0<x<70)):
                player.note_on(int((notes['c1'])),127)
                sleep(0.27)
                player.note_off(int((notes['c1'])),127)
                n = notes['c1']
                
            if ((93<x<139) and (y<250) ) or ((350>y>250)and(75<x<158)):
                player.note_on(int((notes['d1'])),127)
                sleep(0.27)
                player.note_off(int((notes['d1'])),127)
                n = notes['d1']
             
            
            if ((181<x<228) and (y<250) ) or ((350>y>250)and(161<x<230)):
                player.note_on(int((notes['e1'])),127)
                sleep(0.27)
                player.note_off(int((notes['e1'])),127)
                n = notes['e1']
                
            if ((233<x<285) and (y<250) ) or ((350>y>250)and(233<x<307)):
                player.note_on(int((notes['f1'])),127)
                sleep(0.27)
                player.note_off(int((notes['f1'])),127)
                n = notes['f1']
            
            if ((328<x<375) and (y<250) ) or ((350>y>250)and(310<x<393)):
                player.note_on(int((notes['g1'])),127)
                sleep(0.27)
                player.note_off(int((notes['g1'])),127)
                n = notes['g1']
            
            
            if ((415<x<463) and (y<250) ) or ((350>y>250)and(397<x<483)):
                player.note_on(int((notes['a1'])),127)
                sleep(0.27)
                player.note_off(int((notes['a1'])),127)
                n = notes['a1']
            
            if ((505<x<564) and (y<250) ) or ((350>y>250)and(488<x<564)):
                player.note_on(int((notes['b1'])),127)
                sleep(0.27)
                player.note_off(int((notes['b1'])),127)
                n = notes['b1']
                
            if ((568<x<626) and (y<250) ) or ((350>y>250)and(568<x<647)):
                player.note_on(int((notes['c2'])),127)
                sleep(0.27)
                player.note_off(int((notes['c2'])),127)
                n = notes['c2']
                
            if ((670<x<714) and (y<250) ) or ((350>y>250)and(650<x<732)):
                player.note_on(int((notes['d2'])),127)
                sleep(0.27)
                player.note_off(int((notes['d2'])),127)
                n = notes['d2']
                
            if ((756<x<805) and (y<250) ) or ((350>y>250)and(737<x<804)):
                player.note_on(int((notes['e2'])),127)
                sleep(0.27)
                player.note_off(int((notes['e2'])),127)
                n = notes['e2']
                
            if ((807<x<863) and (y<250) ) or ((350>y>250)and(807<x<880)):
                player.note_on(int((notes['f2'])),127)
                sleep(0.27)
                player.note_off(int((notes['f2'])),127)
                n = notes['f2']
            if ((904<x<949) and (y<250) ) or ((350>y>250)and(886<x<968)):
                player.note_on(int((notes['g2'])),127)
                sleep(0.27)
                player.note_off(int((notes['g2'])),127)
                n = notes['g2']
                
            if ((991<x<1038) and (y<250) ) or ((350>y>250)and(972<x<1059)):
                player.note_on(int((notes['a2'])),127)
                sleep(0.27)
                player.note_off(int((notes['a2'])),127)
                n = notes['a2']
                
            if ((1080<x<1149) and (y<250) ) or ((350>y>250)and(1062<x<1149)):
                player.note_on(int((notes['b2'])),127)
                sleep(0.27)
                player.note_off(int((notes['b2'])),127)
                n = notes['b2']
            #############################################
                
            if ((53<x<92) and (y<250)):
                player.note_on(int((notes['c#1'])),127)
                sleep(0.27)
                player.note_off(int((notes['c#1'])),127)
                n = notes['c#1']
                           
            if ((140<x<180) and (y<250)):
                player.note_on(int((notes['d#1'])),127)
                sleep(0.27)
                player.note_off(int((notes['d#1'])),127)
                n = notes['d#1']
                           
            if ((287<x<327) and (y<250)):
                player.note_on(int((notes['f#1'])),127)
                sleep(0.27)
                player.note_off(int((notes['f#1'])),127)
                n = notes['f#1']
                                
            if ((376<x<414) and (y<250)):
                player.note_on(int((notes['g#1'])),127)
                sleep(0.27)
                player.note_off(int((notes['g#1'])),127)
                n = notes['g#1']
                           
            if ((466<x<503) and (y<250)):
                player.note_on(int((notes['a#1'])),127)
                sleep(0.27)
                player.note_off(int((notes['a#1'])),127)
                n = notes['a#1']
                            
            if ((629<x<666) and (y<250)):
                player.note_on(int((notes['c#2'])),127)
                sleep(0.27)
                player.note_off(int((notes['c#2'])),127)
                n = notes['c#2']  
                          
            if ((716<x<754) and (y<250)):
                player.note_on(int((notes['d#2'])),127)
                sleep(0.27)
                player.note_off(int((notes['d#2'])),127)
                n = notes['d#2'] 
                          
            if ((865<x<902) and (y<250)):
                player.note_on(int((notes['f#2'])),127)
                sleep(0.27)
                player.note_off(int((notes['f#2'])),127)
                n = notes['f#2']
                           
            if ((951<x<989) and (y<250)):
                player.note_on(int((notes['g#2'])),127)
                sleep(0.27)
                player.note_off(int((notes['g#2'])),127)
                n = notes['g#2']    
                          
            if ((1041<x<1079) and (y<250)):
                player.note_on(int((notes['a#2'])),127)
                sleep(0.27)
                player.note_off(int((notes['a#2'])),127)
                n = notes['a#2']
                          
            return n
##########################################################################
##########################################################################
def intro():
    
    pygame.font.init() # you have to call this at the start, 
                   # if you want to use this module.
    myfont = pygame.font.SysFont('Comic Sans MS', 30)
    dis_with = 1150
    dis_height = 425

    black = (15,15,15)
    white = (255,255,255)
    red = (255,0,0)
    gray = (180,180,180)
    blue = (0,0,255)
    gamedisplay = pygame.display.set_mode((dis_with, dis_height))

    pygame.display.set_caption("AI POWERED KEYBOARD")
    
    pygame.display.update()
    pygame.draw.rect(gamedisplay, gray, (0,0,  80,350))  
    pygame.draw.rect(gamedisplay, gray, (80,0, 100,350)) 
    pygame.draw.rect(gamedisplay, gray, (170,0,80,350)) 
    pygame.draw.rect(gamedisplay, gray, (240,0,80,350)) 
    pygame.draw.rect(gamedisplay, gray, (310,0,105,350)) 
    pygame.draw.rect(gamedisplay, gray, (405,0,100,350)) 
    pygame.draw.rect(gamedisplay, gray, (495,0,80,350)) 
    
    pygame.draw.rect(gamedisplay, gray, (575,0,80,350))  
    pygame.draw.rect(gamedisplay, gray, (655,0,100,350)) 
    pygame.draw.rect(gamedisplay, gray, (745,0,80,350)) 
    pygame.draw.rect(gamedisplay, gray, (815,0,80,350)) 
    pygame.draw.rect(gamedisplay, gray, (885,0,105,350)) 
    pygame.draw.rect(gamedisplay, gray, (980,0,100,350)) 
    pygame.draw.rect(gamedisplay, gray, (1070,0,80,350)) 
    
#--------------------------------------------------------    
    pygame.draw.rect(gamedisplay, black, (53, 0,40,250))  
    pygame.draw.rect(gamedisplay, black, (140,0,40,250)) 
    pygame.draw.rect(gamedisplay, black, (288,0,40,250)) 
    pygame.draw.rect(gamedisplay, black, (375,0,40,250)) 
    pygame.draw.rect(gamedisplay, black, (465,0,40,250)) 
    
    pygame.draw.rect(gamedisplay, black, (628, 0,40,250))  
    pygame.draw.rect(gamedisplay, black, (715,0,40,250)) 
    pygame.draw.rect(gamedisplay, black, (863,0,40,250)) 
    pygame.draw.rect(gamedisplay, black, (950,0,40,250)) 
    pygame.draw.rect(gamedisplay, black, (1040,0,40,250)) 

#--------------------------------------------------------  
    pygame.draw.rect(gamedisplay, black, (19+53,0,3,350))
    pygame.draw.rect(gamedisplay, black, (19+140,0,3,350))
    pygame.draw.rect(gamedisplay, black, (230,0,3,350))    
    pygame.draw.rect(gamedisplay, black, (19+288,0,3,350))
    pygame.draw.rect(gamedisplay, black, (19+375,0,3,350))
    pygame.draw.rect(gamedisplay, black, (19+465,0,3,350))
    
    pygame.draw.rect(gamedisplay, black, (565,0,3,350)) 
        
    pygame.draw.rect(gamedisplay, black, (19+628,0,3,350))
    pygame.draw.rect(gamedisplay, black, (19+715,0,3,350))
    pygame.draw.rect(gamedisplay, black, (805,0,3,350)) 
    pygame.draw.rect(gamedisplay, black, (19+863,0,3,350))
    pygame.draw.rect(gamedisplay, black, (19+950,0,3,350))
    pygame.draw.rect(gamedisplay, black, (19+1040,0,3,350))
    
    pygame.draw.rect(gamedisplay, red, (0,350,160,75))
    pygame.draw.rect(gamedisplay, blue, (160,350,160,75))
    
    textsurface1 = myfont.render('RESET', False, (0, 255, 0))
    textsurface2 = myfont.render('DONE', False, (0, 255, 0))
    gamedisplay.blit(textsurface1, (20,365))
    gamedisplay.blit(textsurface2, (180,365))
#--------------------------------------------------------  
    pygame.display.update()

##########################################################################
##########################################################################

num_units = [1024,1024]
num_time_steps = None
num_inputs = 1

num_outputs =1




X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
Y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])



cells = [BasicLSTMCell(num_units=n) for n in num_units]
stacked_rnn_cell = MultiRNNCell(cells)
cell = tf.contrib.rnn.OutputProjectionWrapper(stacked_rnn_cell, output_size = num_outputs)


outputs, states = tf.nn.dynamic_rnn(cell, X, dtype = tf.float32)

saver = tf.train.Saver()


def predict(x):
    lenth = len(x)
    if lenth <1:
        raise ValueError("A lenth <= to 0 is not valid in this definition")
        
    with tf.Session() as sess:
        saver.restore(sess, "./train_ckpt/train" )
        preds = []
        topr = x 
        for i in range(lenth):
         
            X_new = np.reshape(topr, (1, len(topr), 1))
            y_pred = np.reshape((sess.run(outputs, feed_dict = {X:X_new})), (len(topr),))
           
            preds.append(y_pred[-1])
          
            topr = y_pred
         
         
    return playlist([0]+preds)
    
def playlist(plist):
    for i in plist:
    
        player.note_on(int(noted(i)), 127)
        sleep(0.27)
        player.note_off(int(noted(i)), 127)
        
prednotes = []        
intro()


while True:
    asd = pygame.event.get()
    if len(asd)>0:
        event = asd[0]
    
        if event.type==QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            if (x < 160) and (y>350):
                prednotes = []
            if (164 < x < 320) and (y>350):
                print(prednotes)
                predict(prednotes)
                prednotes = []
                
                          
            note = freq(playnote(x,y))
           # note = freq(playnoteardu(x,y))
            prednotes.append(note)
                
                
            print((x,y))
                
    sleep(0.06)

           
#%reset
