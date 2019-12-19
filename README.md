# LSTM_Music_Making_Machine
Using Long short-term memory (LSTM) Reccurent Neural Networks to create a musical response to a tune played on a virtual piano. Coded with Python3 and Tensorflow.

#

You must use anaconda for this program, or install the libaries with only pip.

Before running the code, please install the following libaries with pip and conda:

1. Tensorflow: conda install tensorflow   (If you have an NVIDIA GPU: conda install tensorflow-gpu)

2. Keras: conda install keras

3. Pygame: conda install -c cogsci pygame

4. MIDO : pip install mido

#

We will be using a 2 layer LSTM neural network to create musical responces to tune played by a user on a virtual piano. This code is built in python 3.7, and uses Tensorflow and Keras for the model architecture. 

It is reccomended to train this model with a NVIDIA GPU that suppord CUDA for a faster training experience. It took 3 - 4 days of training to achieve its full potential on a GTX1080, but it will work with just 24 hours too. 

#

To train the program, use the train.py file. This will create a training checkpoint that you can use in the piano.py program, where pygame will be used to create a virtual piano. Once you have finished playing the tune on the piano, press done for the machine learning model to play the musical response.

#

A pretrained model is available on request, as it is to big to place on Github (150 mb). Once a checkpoint file for the model is created, you can run the piano.py on a CPU


Read more about the Music Making Machine project and see it in action at: https://hotpoprobot.com/2019/07/26/making-music-using-machine-learning-music-making-machine-m3/
