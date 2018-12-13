"""
Created on Tue Nov 27 20:40:11 2018

@author: Fanjie Kong
"""
import sys
sys.path.append('D:/桌面文件2/出国准备/Duke/BME503-Neuroc/Final_Project/My_final_project/pattern_bp/src/')
from brian2 import *
import numpy as np
from models.snn_ReSuMe import SpikingNeuralNetwork
# from convergable_datasets import *
from convergable_datasets import *
from minist_dataset_test import input_indices_0, input_spike_times_0, input_indices_9, input_spike_times_9, spike_train1, spike_train2, bunch_of_zero, bunch_of_nine
# The parameters are claimed in the data_and_paras
structure = [1, 40, 1]
Tmax = 350
trainer = SpikingNeuralNetwork(seed=1, nn_architecture=structure, input_type='spike_train')
trainer.train_config(mode='no_bp',
                     eqs=eqs,
                     threshold=threshold,
                     class_num=1,
                     refractory=4*ms,
                     reset=reset,
                     syn_eqs=syn_eqs,
                     on_pre=on_pre,
                     gmax=gmax,
                     Er=Er,
                     lr=lr,
                     tau=tau,
                     A=A,
                     R=R)

# Xor datasets
trainer.train(input_data=[[input_indices_0, input_spike_times_0]],
              target=[spike_train1],
              interval=Tmax,
              method='simultaneously_pattern_bp',
              max_epochs=50, # epoch for each_layer
              learning_rate=1,
              decay_rate=0.95,
              decay_epoch=30,
              show_output=True,
              save_weights=True,
              save_name= '0_weights.npy')
similarity_list_1 = trainer.predict_figure(bunch_of_zero, spike_train1, load_model_path='0_weights.npy')
similarity_list_2 = trainer.predict_figure(bunch_of_nine, spike_train1, load_model_path='0_weights.npy')

plt.plot(range(len(similarity_list_1)), similarity_list_1, label='Similarity of digits image with label 0')
plt.plot(range(len(similarity_list_2)), similarity_list_2, label='Similarity of digits image with label 9')
plt.ylabel('Similarity')
plt.xlabel('Number of samples')
plt.legend()
plt.savefig('Similarity_curve_digits_recognition.jpg')

# prediction = trainer.predict([[input_indices_1, input_spike_train_1], [input_indices_2, input_spike_train_2],
#                           [input_indices_3, input_spike_train_3], [input_indices_4, input_spike_train_4]])
# print('Test result: ', prediction)
# print('Predict Label: ', np.argmax(prediction, axis=0))
