"""
Created on Tue Nov 27 20:40:11 2018

@author: Fanjie Kong
"""
import sys
sys.path.append('D:/桌面文件2/出国准备/Duke/BME503-Neuroc/Final_Project/My_final_project/pattern_bp/src/')
from brian2 import *
import numpy as np
from models.snn_ReSuMe import SpikingNeuralNetwork
from data_and_paras import *
# The parameters are claimed in the data_and_paras

trainer = SpikingNeuralNetwork(seed=1, nn_architecture=structure, input_type='spike_train')
trainer.train_config(mode='no_bp',
                     eqs=eqs,
                     threshold=threshold,
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
trainer.train(input_data=[[input_indices, input_spike_train], [input_indices_2, input_spike_train_2]],
              target=[targOut, targOut_2],
              interval=Tmax,
              method='',
              max_epochs=30,
              learning_rate=1,
              decay_rate=0.95,
              decay_epoch=30,
              show_output=True,
              save_weights=True)
prediction = trainer.predict([[input_indices, input_spike_train], [input_indices_2, input_spike_train_2]])
print('Test result: ', prediction)
