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
                     Tmax=Tmax,
                     gmax=gmax,
                     Er=Er,
                     lr=lr,
                     tau=tau,
                     Tw=Tw,
                     A=A,
                     R=R)

trainer.train(input_data=[input_indices, input_spike_train],
              target=targOut,
              max_epochs=300,
              learning_rate=1,
              decay_rate=0.9,
              decay_epoch=30,
              show_output=True,
              save_weights=True)