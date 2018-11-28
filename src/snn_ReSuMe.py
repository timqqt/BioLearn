"""
Created on Tue Nov 27 20:40:11 2018

@author: Fanjie Kong
"""
import scipy.io as sio
import os.path as op
import brian2 as br
import numpy as np
import math as ma
import scipy
import pudb
import sys
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FC
from matplotlib.pyplot import plot, show
import matplotlib.pyplot as plt
from subprocess import call

br.prefs.codegen.target = 'weave'  # use the Python fallback


class SpikingNeuralNetwork:

    def __init__(self, hidden=5, output=2, inputs=3, subc=3, delay=11, seed=None, nn_architecture=None, neuron_eqs=None, input_type='spike_train'):
        # pudb.set_trace()
        self.changes = []
        self.trained = False
        self.rb = 1.0
        self.r = 10.0
        self.dta = 0.2 * br.ms
        # self.delay = delay
        # self.N_inputs = inputs
        # self.N_hidden = hidden
        # self.N_output = output
        # self.N_subc = subc
        self.tauLP = 5.0
        # self.tauIN = 5.0
        self.seed = seed
        np.random.seed(self.seed)
        # self.a, self.d = None, None
        # self.a_post, self.d_post = [], []
        # self.a_pre, self.d_pre = [], []
        # self.data, self.labels = None, None
        self.T = 50
        self.eqs = neuron_eqs
        self.architecture = nn_architecture
        # self.__groups()  # initialize all the network use brian2
        self.input_type = input_type
        self.net = br.Network()

    def no_bp_multilayer_network(self):
        if self.input_type == 'spike_train':
            input_indices = self.input_data[0]
            input_times = self.input_data[1]
            input_nums = self.architecture[0]
            input_X = br.SpikeGeneratorGroup(input_nums, input_indices, input_times, name='input')
        # initialize hidden layer
        hidden_layer = []
        for num, n_hidden in enumerate(self.architecture[1:-1]):
            hidden_layer.append(br.NeuronGroup(n_hidden, self.eqs, clock=br.Clock(br.defaultclock.dt),
                                               threshold=self.threshold, reset=self.reset, refractory=self.refractory,
                                                method='linear', name='hidden_{}'.format(num+1)))
        # output layer
        output_layer = br.NeuronGroup(self.architecture[-1], self.eqs, clock=br.Clock(br.defaultclock.dt),
                                      threshold=self.threshold, reset=self.reset, refractory=self.refractory,
                                        method='linear', name='output')

        # add into network
        self.net.add(input_X)
        self.net.add(hidden_layer)
        self.net.add(output_layer)

        pass

    def train_config(self, mode='no_bp', input_data=None, target=None, threshold='v>-55*mV', reset='v=Er', refractory=4 *br. ms):
        self.input_data = input_data
        self.target = target
        # initialize the parameters of neurons
        self.threshold = threshold
        self.reset = reset
        self.refractory = refractory
        if mode == 'no_bp':
            self.no_bp_multilayer_network()
        pass

    def structure_visulization(self):
        pass

    def data_converter(self):
        pass

    def train(self):
        pass
