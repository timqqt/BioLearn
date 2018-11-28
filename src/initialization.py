"""
Created on Tue Nov 27 20:40:11 2018

@author: Fanjie Kong
"""

from brian2 import *
import numpy as np
from snn_ReSuMe import SpikingNeuralNetwork
# proposed parameters
structure = [6, 6, 1]  # The first number means the number of inputs and the final one means the number of output
A=0.000002
Tw=np.exp(-np.arange(0,11)/5.0)

# Input & output data
targOut=[[22.0, 55.0, 72.0, 90.0, 144.0, 180.0],
         [22.0, 55.0, 90.0, 121.0, 154.0, 187.0],
         [  22.,  180.,  162.,   90.,   72.,   54.,  144.,   36.],
         [22.0, 55.0, 90.0, 121.0, 154.0, 187.0],
         [22.0, 55.0, 90.0, 121.0, 154.0, 187.0]]

i1=arange(17.0,200.0,16.0)
i2=arange(10.9,200,11.0)
i3=arange(8.5,200.0,8.2)
i4=arange(17.9,200.0,18.0)
i5=arange(10.0,200,11.0)
i6=arange(8.0,200.0,8.2)

input_spike_train = np.concatenate((i1, i2, i3, i4, i5, i6),
                                   axis=0, out=None)
input_spike_train = input_spike_train*ms
a1=repeat(0,len(i1),axis=0)
a2=repeat(1,len(i2),axis=0)
a3=repeat(2,len(i3),axis=0)
a4=repeat(3,len(i4),axis=0)
a5=repeat(4,len(i5),axis=0)
a6=repeat(5,len(i6),axis=0)
input_indices = numpy.concatenate((a1, a2, a3, a4, a5, a6),
                                  axis=0, out=None)

eqs='''
dv/dt = (-(v-Er)+R*Iinj)/tau : volt
Iinj:amp
'''
threshold = 'v>-55*mV'
reset = 'v=Er'

trainer = SpikingNeuralNetwork(seed=1, nn_architecture=structure, neuron_eqs=eqs, input_type='spike_train')
trainer.train_config(mode='no_bp', input_data=[input_indices, input_spike_train], target=targOut, threshold='v>-55*mV', reset='v=Er', refractory=4*ms)