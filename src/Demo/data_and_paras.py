import numpy as np
from brian2 import *

# proposed parameters
structure = [6, 6, 1]  # The first number means the number of inputs and the final one means the number of output
Tmax = 200
A = 0.000002
# Tw = np.exp(-np.arange(0, 11)/5.0)

tau = 10*ms
Er = -65*mV
R = 10000000*ohm

gmax = 7.5*10**-6
lr = gmax/90
# Input & output data
# [25.0*float(i+1) for i in range(Tmax//25-1)],
targOut = [[162.0, 90.0, 36.0],
        [178, 145, 128, 110, 56, 20 ],
        [22.0, 55.0, 72.0, 90.0, 144.0, 180.0],
           [22.0, 55.0, 90.0, 180.0, 135.0],
        [22.0, 180., 162.0,90.0,  72.0, 54.,144., 36., 44, 120, 153, 67],
        [144.0, 162.0, 120.0, 180.0],
        [162.0, 90.0, 36.0],
        [22.0, 180., 162.0,90.0,  72.0, 54.,144., 36.],
         [22.0, 55.0, 72.0, 90.0, 144.0, 180.0],
         [22.0, 55.0, 90.0, 121.0, 154.0, 187.0],
         [22.0, 55.0, 90.0, 121.0, 154.0, 187.0],
         [22.0, 55.0, 90.0, 121.0, 154.0, 187.0]]
targOut_2 = [arange(15, 200, 13)]
# frame work for Xor problem
i1=arange(34.0,200.0,99.0)
i2=arange(90.9,200,86.0)
i3=arange(38.5,200.0,92)
i4=arange(162.9,200.0,99.0)
i5=arange(160.0,200,81.0)
i6=arange(88.0,200.0,102.2)

i1_2=arange(17.0,200.0,15.0)
i2_2=arange(15, 200, 13)
i3_2=arange(8.5,200.0,8.2)
i4_2=arange(17.9,200.0,18.0)
i5_2=arange(15, 200, 13)
i6_2=arange(8.0,200.0,8.2)

input_spike_train = np.concatenate((i1, i2, i3, i4, i5, i6),
                                   axis=0, out=None)
input_spike_train_2 = np.concatenate((i1_2, i2_2, i3_2, i4_2, i5_2, i6_2),
                                   axis=0, out=None)
# input_spike_train = np.concatenate((targOut[0], targOut[4], targOut[6], targOut[5], targOut[7], targOut[8]),
#                                     axis=0, out=None)

input_spike_train = input_spike_train*ms
input_spike_train_2 = input_spike_train_2*ms

a1=repeat(0,len(i1),axis=0)
a2=repeat(1,len(i2),axis=0)
a3=repeat(2,len(i3),axis=0)
a4=repeat(3,len(i4),axis=0)
a5=repeat(4,len(i5),axis=0)
a6=repeat(5,len(i6),axis=0)

a1_2=repeat(0,len(i1_2),axis=0)
a2_2=repeat(1,len(i2_2),axis=0)
a3_2=repeat(2,len(i3_2),axis=0)
a4_2=repeat(3,len(i4_2),axis=0)
a5_2=repeat(4,len(i5_2),axis=0)
a6_2=repeat(5,len(i6_2),axis=0)
# a1=repeat(0,len(targOut[0]),axis=0)
# a2=repeat(1,len(targOut[4]),axis=0)
# a3=repeat(2,len(targOut[6]),axis=0)
# a4=repeat(3,len(targOut[5]),axis=0)
# a5=repeat(4,len(targOut[7]),axis=0)
# a6=repeat(5,len(targOut[8]),axis=0)
input_indices = numpy.concatenate((a1, a2, a3, a4, a5, a6),
                                  axis=0, out=None)
input_indices_2 = numpy.concatenate((a1_2, a2_2, a3_2, a4_2, a5_2, a6_2),
                                  axis=0, out=None)

eqs='''
dv/dt = (-(v-Er))/tau : volt
tau : second
Er : volt
'''
threshold = 'v>-55*mV'
reset = 'v=Er'

# The equation for updating synaptic weights
syn_eqs = 'w : 1'
on_pre='v_post+=10**9*w*mV'