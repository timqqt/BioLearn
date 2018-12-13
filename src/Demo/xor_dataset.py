import numpy as np
from brian2 import *
import numpy

# proposed parameters
structure = [3, 40, 2]  # The first number means the number of inputs and the final one means the number of output
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
targOut_1 = [[100], [80]]
targOut_2 = [[80], [100]]
targOut_3 = [[80], [100]]
targOut_4 = [[100], [80]]
# frame work for Xor problem
i1_1=[70]
i2_1=[70]
i3_1=[90]


i1_2=[70]
i2_2=[90]
i3_2=[90]

i1_3=[90]
i2_3=[70]
i3_3=[90]

i1_4=[90]
i2_4=[90]
i3_4=[90]

input_spike_train_1 = np.concatenate((i1_1, i2_1, i3_1),
                                   axis=0, out=None)
input_spike_train_2 = np.concatenate((i1_2, i2_2, i3_2),
                                   axis=0, out=None)
input_spike_train_3 = np.concatenate((i1_3, i2_3, i3_3),
                                   axis=0, out=None)
input_spike_train_4 = np.concatenate((i1_4, i2_4, i3_4),
                                   axis=0, out=None)
# input_spike_train = np.concatenate((targOut[0], targOut[4], targOut[6], targOut[5], targOut[7], targOut[8]),
#                                     axis=0, out=None)

input_spike_train_1 = input_spike_train_1*ms
input_spike_train_2 = input_spike_train_2*ms
input_spike_train_3 = input_spike_train_3*ms
input_spike_train_4 = input_spike_train_4*ms

a1_1=repeat(0,len(i1_1),axis=0)
a2_1=repeat(1,len(i2_1),axis=0)
a3_1=repeat(2,len(i3_1),axis=0)


a1_2=repeat(0,len(i1_2),axis=0)
a2_2=repeat(1,len(i2_2),axis=0)
a3_2=repeat(2,len(i3_2),axis=0)


a1_3=repeat(0,len(i1_3),axis=0)
a2_3=repeat(1,len(i2_3),axis=0)
a3_3=repeat(2,len(i3_3),axis=0)


a1_4=repeat(0,len(i1_4),axis=0)
a2_4=repeat(1,len(i2_4),axis=0)
a3_4=repeat(2,len(i3_4),axis=0)

# a1=repeat(0,len(targOut[0]),axis=0)
# a2=repeat(1,len(targOut[4]),axis=0)
# a3=repeat(2,len(targOut[6]),axis=0)
# a4=repeat(3,len(targOut[5]),axis=0)
# a5=repeat(4,len(targOut[7]),axis=0)
# a6=repeat(5,len(targOut[8]),axis=0)
input_indices_1 = numpy.concatenate((a1_1, a2_1, a3_1),
                                  axis=0, out=None)
input_indices_2 = numpy.concatenate((a1_2, a2_2, a3_2),
                                  axis=0, out=None)
input_indices_3 = numpy.concatenate((a1_3, a2_3, a3_3),
                                  axis=0, out=None)
input_indices_4 = numpy.concatenate((a1_4, a2_4, a3_4),
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