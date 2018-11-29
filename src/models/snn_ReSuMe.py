"""
Created on Tue Nov 27 20:40:11 2018

@author: Fanjie Kong
"""

import numpy as np
import brian2 as br
import scipy.io as sio
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian1DKernel, convolve

# time setting  --may be free?
br.defaultclock.dt=0.1*br.ms
dt = br.defaultclock.dt/br.ms


class SpikingNeuralNetwork:

    def __init__(self, seed=None, nn_architecture=None, input_type='spike_train'):
        '''
        initialization function
        :param seed: int
                random numbber seed
        :param nn_architecture: list
                architecture of the neural network
        :param neuron_eqs: string
                equations for neurons
        :param input_type: string
                input type, numerical input or sequential input
        '''
        self.alpha = 1
        self.decay_rate = None
        self.decay_epoch = None
        self.seed = seed
        np.random.seed(self.seed)
        self.changes = []
        self.trained = False
        self.rb = 1.0
        self.r = 10.0
        self.dta = 0.2 * br.ms
        self.tauLP = 5.0
        self.T = 50
        self.eqs = None
        self.architecture = nn_architecture
        self.input_type = input_type
        self.net = br.Network()
        # initialize the parameters of neurons
        self.parameters = {}
        self.input_data = None
        self.target = None
        self.threshold = None
        self.reset = None
        self.refractory = None

    def no_bp_multilayer_network(self):
        '''
        Build up the network
        :return: None
        '''

        if self.input_type == 'spike_train':
            input_nums = self.architecture[0]
            # we will re-initialize the input spike trains when training
            input_X = br.SpikeGeneratorGroup(input_nums, [0], [0]*br.ms, name='input')
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

        # initialize synaptic connections -- use dense connections
        I_to_H = br.Synapses(self.net['input'], self.net['hidden_1'], self.syn_eqs,
                             on_pre=self.on_pre, name='syn_I2H')
        # sparse connect -- maybe encounter the problem he says
        # I_to_H.connect(i=[i for i in range(self.architecture[0])], j=[j for j in range(self.architecture[1])])
        I_to_H.connect('True')
        if len(hidden_layer) > 1:  # if multi-layer, not applicable now

            pass

        H_to_O = br.Synapses(self.net['hidden_'+ str(len(hidden_layer))], self.net['output'], self.syn_eqs
                             , on_pre=self.on_pre, name='syn_H2O')
        H_to_O.connect('True')  # Dense connection

        self.net.add(I_to_H)
        # add more hidden layers connection
        self.net.add(H_to_O)

        pass

    def parameters_initialization(self):
        '''
        initialize parameters in the network
        :return: None
        '''
        # initialize hidden
        for i in range(len(self.architecture[1:-1])):
            self.net['hidden_' + str(i + 1)].Er = self.parameters['Er']
            self.net['hidden_' + str(i + 1)].tau = self.parameters['tau']
            self.net['hidden_'+str(i+1)].v = self.parameters['Er']
        # initialize output
        self.net['output'].Er = self.parameters['Er']
        self.net['output'].tau = self.parameters['tau']
        self.net['output'].v = self.parameters['Er']

        # initialize synapse
        np.random.standard_normal()
        w1 = self.net['syn_I2H'].w
        w2 = self.net['syn_H2O'].w
        self.net['syn_I2H'].w = self.parameters['gmax']
        self.net['syn_H2O'].w = self.parameters['gmax']

    def monitor_initialization(self):
        '''
        initialize the monitor for the network
        :return: None
        '''
        self.net.add(br.SpikeMonitor(self.net['input'], name='monitor_input'))
        print(len(self.architecture[1:-1]))
        for i in range(len(self.architecture[1:-1])):
            self.net.add(br.SpikeMonitor(self.net['hidden_'+str(i+1)], name='monitor_hidden_'+str(i+1)))
        self.net.add(br.SpikeMonitor(self.net['output'], name='monitor_output'))

    def train_config(self, eqs = None, mode='no_bp', threshold='v>-55*mV', reset='v=Er', refractory=4 *br. ms, syn_eqs=None, on_pre=None,**kwargs):
        '''

        :param mode: string
                the mode of network
        :param threshold: string
                threshold for neurons to reset
        :param reset: string
                reset equation
        :param refractory: brian quantity
                refractory times for neuron
        :param kwargs:
                Any other parameters
        :return: None
        '''
        # initialize the parameters of neurons
        self.eqs = eqs
        self.threshold = threshold
        self.reset = reset
        self.refractory = refractory
        self.parameters = kwargs
        self.syn_eqs = syn_eqs
        self.on_pre = on_pre
        if mode == 'no_bp':
            # construct the model
            self.no_bp_multilayer_network()
            self.parameters_initialization()
            self.monitor_initialization()
        pass

    def structure_visulization(self):
        pass

    def data_converter(self):
        pass

    def train(self, input_data, target, max_epochs=100, learning_rate=1, decay_rate=None, decay_epoch=None, show_output=False, save_weights=False):
        '''
        :param input_data: list
                [input indices, input times]
        :param target: list
                [target spike trains]
        :param max_epochs: int
                max training epochs
        :param learning_rate: float
                learning rate
        :param decay_rate: float
                decay rate of learning rate
        :param decay_epoch: int
                decay_epoch of learning rate
        :param show_figure: bool
                if True, show the figure of output spikes and desired spikes during training
        :param save_weights: bool
                if True, save the weights of neural network
        :return: None
        '''
        # load learning rate parameters
        self.alpha = learning_rate
        self.decay_rate = decay_rate
        self.decay_epoch = decay_epoch
        # load input and target
        self.input_data = input_data
        self.target = target
        self.net['input'].set_spikes(self.input_data[0], self.input_data[1])
        # now just for patter matching version
        Tmax = self.parameters['Tmax']
        sd = np.zeros((int(Tmax / dt) + 1, self.architecture[-1]))
        so = np.zeros((int(Tmax / dt) + 1, self.architecture[-1]))

        for j in range(self.architecture[-1]):
            for i in range(0, int(Tmax / dt) + 1):
                if dt * i in self.target[j][:]:
                    sd[i, j] = 1
                else:
                    sd[i, j] = 0

        #
        P = 0.01
        dwdt = np.zeros((self.architecture[1], self.architecture[-1]))
        self.net.store()
        stored_IHw = br.Quantity(self.net['syn_I2H'].w, copy=True) # Thoes are constant
        stored_w = br.Quantity(self.net['syn_H2O'].w, copy=True)

        P_total = np.zeros((self.architecture[-1], max_epochs))
        initial = 1
        Weighttotal = []
        Weighttotal2 = []
        testindex = 0

        for epoch in range(max_epochs):
            if self.decay_rate is not None:
                if epoch % decay_epoch == 0:
                    self.alpha *= self.decay_rate
            if min(P_total[:, testindex - 1]) >= 0.95:
                break
            self.net['syn_I2H'].w[:, :] = stored_IHw
            self.net['syn_H2O'].w[:, :] = stored_w
            Weighttotal.append(np.array(self.net['syn_H2O'].w))
            Weighttotal2.append(np.array(self.net['syn_I2H'].w))
            self.net.run(Tmax * br.ms)
            if initial == 1:
                spikeinitial_out = np.array(self.net['monitor_output'].i)
                spikeinitial_out_t = np.array(self.net['monitor_output'].t)
                initial = 0

            for pattern in range(self.architecture[-1]):

                P = self.compute_error(self.net['monitor_output'], sd[:, pattern], pattern, so[:, pattern])
                P_total[pattern, testindex] = P
                # input spikes into bit stream
                for i in range(self.architecture[1]):

                    spike_hidden = self.net['monitor_hidden_1'].spike_trains()
                    sh = np.zeros((int(Tmax / dt) + 1, self.architecture[-1]))
                    for ti in spike_hidden[i]:
                        sh[int(float(ti) * 1000 / 0.1), pattern] = 1

                    spike_trains_out = self.net['monitor_output'].spike_trains()
                    so = np.zeros((int(Tmax / dt) + 1, self.architecture[-1]))
                    for ti in spike_trains_out[pattern]:
                        so[int(float(ti) * 1000 / 0.1), pattern] = 1

                    conv = convolve(self.parameters['A'] * sh[:, pattern], self.parameters['Tw'])
                    dwdt[i, pattern] = sum((1.0 / self.architecture[1]) * (
                        self.parameters['lr'] * (sd[:, pattern] - so[:, pattern]) + (sd[:, pattern] - so[:, pattern]) * conv))
                    self.net['syn_H2O'].w[i, pattern] = self.net['syn_H2O'].w[i, pattern] + self.alpha * dwdt[i, pattern]

            print('update ', testindex, ' finished')

            if show_output:
                self.show_figure()
            stored_w = br.Quantity(self.net['syn_H2O'].w, copy=True)
            stored_IHw = br.Quantity(self.net['syn_I2H'].w, copy=True)
            testindex = testindex + 1
            self.net.restore()

        spikefinal_out = np.array(self.net['monitor_output'].i)
        spikefinal_out_t = np.array(self.net['monitor_output'].t)
        if save_weights:
            sio.savemat('outputsaved', {'Weighttotal': Weighttotal, 'spikeinitial_out_t': spikeinitial_out_t,
                                    'spikeinitial_out': spikeinitial_out, 'spikefinal_out_t': spikefinal_out_t,
                                    'spikefinal_out': spikefinal_out, 'P_total': P_total})
        br.plot(Weighttotal)
        br.title('Performance Curve')
        br.xlabel('epoch')
        br.show()

    def compute_error(self, output, sd, pattern, so):
        spike_out = output.spike_trains()
        # print(spike_out[pattern])

        for ti in spike_out[pattern]:
            so[int(float(ti) * 1000 / 0.1)] = 1

        if len(spike_out[pattern]) == 0:
            return 0
            # Create kernel
        g = np.sqrt(2 * 3.1415926) * 4 * Gaussian1DKernel(stddev=4)

        # Convolve data
        vd = convolve(sd, g)

        vo = convolve(so, g)

        C = np.dot(vd, vo) / (np.linalg.norm(vd) * np.linalg.norm(vo))

        print('return cost for pattern ', pattern + 1)
        print(C)

        return C

    def show_figure(self):
        step = 0.001
        time_series = np.arange(0, self.parameters['Tmax'], step)
        spike_trains_o = np.zeros(len(time_series))
        spike_trains_d = np.zeros(len(time_series))
        for spike_time_o in self.net['monitor_output'].t/br.ms:
            spike_trains_o[int(spike_time_o/step)] = 1
        for spike_time_d in self.target[0]:
            spike_trains_d[int(spike_time_d/step)] = 1

        plt.figure(1)
        plt.subplot(211)
        plt.plot(time_series, spike_trains_o, label='Output spike train')
        plt.plot(time_series, spike_trains_d, label='Desired spike train')
        plt.xlabel('t/ms')
        plt.legend()
        plt.ylim(-0.8, 2.2)
        plt.subplot(212)
        plt.plot(self.net['monitor_output'].t/br.ms, self.net['monitor_output'].i, 'k*', label='Output spike train')
        plt.plot(self.target[0], 0.2*np.ones_like(self.target[0]), 'r*', label='Desired spike train')
        plt.ylim(-.3, 2)
        plt.show()
