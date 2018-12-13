"""
Created on Tue Nov 27 20:40:11 2018

@author: Fanjie Kong
"""
import random
import numpy as np
import brian2 as br
import scipy.io as sio
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian1DKernel, convolve


class SpikingNeuralNetwork:

    def __init__(self, seed=None, nn_architecture=None, dt=0.1, input_type='spike_train'):
        '''
        initialization function
        :param seed: int
                random numbber seed
        :param nn_architecture: list
                architecture of the neural network
        :param dt: float
                time step for simulation
        :param input_type: string
                input type, numerical input or sequential input
        '''
        # time setting
        br.defaultclock.dt = dt * br.ms
        self.dt = br.defaultclock.dt / br.ms
        # initialize parameters
        self.alpha = 1
        self.decay_rate = None
        self.decay_epoch = None
        self.seed = seed
        self.class_num = 0
        np.random.seed(self.seed)
        self.changes = []
        self.trained = False
        self.rb = 1.0
        self.r = 10.0
        self.dta = 0.2 * br.ms
        self.tauLP = 5.0
        self.T = 50
        self.eqs = None
        self.interval = 0
        self.architecture = nn_architecture
        self.input_type = input_type
        self.net = br.Network()
        # initialize the parameters of neurons
        self.parameters = {}
        self.weights_container = {}
        self.input_data = None
        self.target = None
        self.threshold = None
        self.reset = None
        self.refractory = None
        self.stored_w = []
        self.name_list_syns = []
        self.name_list_layers = []

    def no_bp_multilayer_network(self):
        '''
        Build up the network
        :return: None
        '''

        if self.input_type == 'spike_train':
            input_nums = self.architecture[0]
            # we will re-initialize the input spike trains when training
            input_X_sp = br.SpikeGeneratorGroup(input_nums, [0], [0]*br.ms, name='input_sp')
            input_X = br.NeuronGroup(input_nums, self.eqs, clock=br.Clock(br.defaultclock.dt),
                                      threshold=self.threshold, reset=self.reset, refractory=self.refractory,
                                        method='linear', name='input')
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
        self.net.add(input_X_sp)
        self.net.add(input_X)
        self.net.add(hidden_layer)
        self.net.add(output_layer)

        # initialize synaptic connections -- use dense connections
        I_to_I = br.Synapses(self.net['input_sp'], self.net['input'], self.syn_eqs,
                             on_pre=self.on_pre, name='sy_I2I')

        I_to_H = br.Synapses(self.net['input'], self.net['hidden_1'], self.syn_eqs,
                             on_pre=self.on_pre, name='syn_I2H')
        # if multi-layer -> applicable
        if len(self.architecture[1:-1]) > 1:
            H_to_H = []
            hidden_layers_list = []
            for each_obj in self.net.objects:
                if 'hidden_' in each_obj.name:
                    hidden_layers_list.append(each_obj.name)
            for index, each_hl in enumerate(hidden_layers_list[:-1]):
                H_to_H.append(br.Synapses(self.net[hidden_layers_list[index]], self.net[hidden_layers_list[index + 1]], self.syn_eqs,
                             on_pre=self.on_pre, on_post=None, name='syn_H2H_'+ str(index+1)))
                H_to_H[index].connect('True')
            self.net.add(H_to_H)

        # sparse connect -- maybe encounter the problem he says
        I_to_I.connect(i=[i for i in range(self.architecture[0])], j=[j for j in range(self.architecture[0])])
        #I_to_H.connect(i=[i for i in range(self.architecture[0])], j=[j for j in range(self.architecture[1])])
        I_to_H.connect('True')

        H_to_O = br.Synapses(self.net['hidden_' + str(len(hidden_layer))], self.net['output'], self.syn_eqs
                             , on_pre=self.on_pre, name='syn_H2O')
        H_to_O.connect('True')  # Dense connection

        self.net.add(I_to_I)
        self.net.add(I_to_H)
        # add more hidden layers connection
        self.net.add(H_to_O)

    def parameters_initialization(self):
        '''
        initialize parameters in the network
        :return: None
        '''
        # initialize hidden
        self.net['input'].Er = self.parameters['Er']
        self.net['input'].tau = self.parameters['tau']
        self.net['input'].v = self.parameters['Er']
        for i in range(len(self.architecture[1:-1])):
            self.net['hidden_' + str(i + 1)].Er = self.parameters['Er']
            self.net['hidden_' + str(i + 1)].tau = self.parameters['tau']
            self.net['hidden_'+str(i+1)].v = self.parameters['Er']
        # initialize output
        self.net['output'].Er = self.parameters['Er']
        self.net['output'].tau = self.parameters['tau']
        self.net['output'].v = self.parameters['Er']

        # initialize synapse
        # initialize the parameters with Gaussian distribution randomly
        self.net['sy_I2I'].w[:, :] = self.parameters['gmax']
        self.net['syn_I2H'].w[:, :] = self.parameters['gmax']

        # self.net['syn_I2H'].w[:, :] = self.parameters['gmax'] * np.random.normal(loc=1, scale=2, size=self.architecture[0]*self.architecture[1])
        # self.net['syn_H2O'].w[:, :] = self.parameters['gmax'] * np.random.normal(loc=0, size=self.architecture[1])
        self.net['syn_H2O'].w[:, :] = self.parameters['gmax']
        # self.net['syn_H2O'].w[0] = -self.parameters['gmax']
        # self.net['syn_H2O'].w[1] = -self.parameters['gmax']
        # self.net['syn_H2O'].w[2] = 1.5*self.parameters['gmax']

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

    def train_config(self, eqs = None, mode='no_bp', class_num=2, threshold='v>-55*mV', reset='v=Er', refractory=4 *br. ms, syn_eqs=None, on_pre=None,**kwargs):
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
        self.class_num = class_num
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

    def _time_train_to_bit(self, interval, num_neurons, target, output):
        Tmax = interval
        sd = np.zeros((int(Tmax / self.dt) + 1, num_neurons))
        # convert the spike trains to bit trains This can be put into Optimizer

        for j in range(num_neurons):
            for i in range(0, int(Tmax / self.dt) + 1):
                if self.dt * i in target[j][:]:
                    sd[i, j] = 1
                else:
                    sd[i, j] = 0

        spike_trains_out = output.spike_trains()
        so = np.zeros((int(interval / self.dt) + 1, num_neurons))
        for pattern in range(num_neurons):
            for ti in spike_trains_out[pattern]:
                # so[int(float(ti) * 1000 / 0.1), pattern] = 1
                ti = (float(ti) * 1000 / 0.1 <= interval / 0.1) and int(float(ti) * 1000 / 0.1) or int(interval / 0.1)
                so[ti, pattern] = 1

        return sd, so

    def train(self, input_data, target, interval=200, max_epochs=100,
              learning_rate=1, method='pattern_bp', decay_rate=None,
              decay_epoch=None, show_output=False,
              save_weights=False, save_name=None):
        '''
        :param input_data: list
                [input indices, input times]
        :param target: list
                [target spike trains]
        :param interval: float
                the time interval for each epoch
        :param max_epochs: int
                max training epochs
        :param learning_rate: float
                learning rate
        :param method: string
                method to train the network
        :param decay_rate: float
                decay rate of learning rate
        :param decay_epoch: int
                decay_epoch of learning rate
        :param show_output: bool
                if True, show the figure of output spikes and desired spikes during training
        :param save_weights: bool
                if True, save the weights of neural network
        :return: None
        '''
        # load learning rate parameters
        self.alpha = learning_rate
        self.decay_rate = decay_rate
        self.decay_epoch = decay_epoch
        # load input and target should be alternative when training multi-input problem
        self.input_data = input_data
        self.target = target
        # self.net['input_sp'].set_spikes(self.input_data[0], self.input_data[1])
        # now just for patter matching version
        Tmax = interval
        self.interval = interval
        # sd, so = self.time_train_to_bit(interval, self.architecture[-1], self.target)
        self.weights_container = {'syn_I2H': [], 'syn_H2O': []}

        #dwdt = np.zeros((self.architecture[1], self.architecture[-1]))
        self.net.store()
        stored_I2H = br.Quantity(self.net['syn_I2H'].w, copy=True) # Thoes are constant
        stored_H2O = br.Quantity(self.net['syn_H2O'].w, copy=True)

        P_total = np.zeros((self.class_num, np.max(max_epochs)))
        P = 0
        initial = 1
        # now just support one single layer bp
        print('Method is ', method)
        if method == 'pattern_bp':
            target_now = self.target[0][0]  # now we just use one target in this experiment
            sub_target = []  # from left to right finally

            # If num(hidden_layer) > 1, how to program assigner ?
            for index, num_neurons in enumerate(self.architecture[1:-1][::-1]):
                # reverse it and cut the head and tails down (who does not need subsets to learn)
                sub_target.append(self.assigner(num_neurons, target_now))
                target_now = sub_target[index]
            sub_target = sub_target[::-1]  # train from input to output
            # use another train method
            # list(a.keys()) --> find all keys
            # get the name of all layers and synapses, so that we can train one by one
            name_list_layers = []
            name_list_syns = []
            for each_obj in self.net.objects:
                if 'monitor_' in each_obj.name:
                    name_list_layers.append(each_obj.name)
                elif 'syn' in each_obj.name and 'pre' not in each_obj.name:
                    name_list_syns.append(each_obj.name)
            print(name_list_syns)
            print(name_list_layers)
            sub_target.append(self.target[0])
            target_list = sub_target
            # Initialize weights container
            self.stored_w = []
            for each_syn_name in name_list_syns:
                self.stored_w.append(br.Quantity(self.net[each_syn_name].w, copy=True))

            for num_layer in range(len(self.architecture)-1):
                self.alpha = learning_rate # reset the learning rate
                if num_layer > 0:
                    self.alpha *= 20
                for epoch in range(max_epochs[num_layer]):
                    # decay setting
                    if self.decay_rate is not None:
                        if epoch % decay_epoch == 0:
                            self.alpha *= self.decay_rate
                    # now I do not wanna consider multi-target task
                    for cls in range(self.class_num):
                        self.net['input_sp'].set_spikes(self.input_data[cls][0], self.input_data[cls][1])
                        if self.decay_rate is not None:
                            if epoch % decay_epoch == 0:
                                self.alpha *= self.decay_rate
                        # If converged, stop it
                        if min(P_total[:, epoch - 1]) >= 0.95:
                            break
                        # Get out the stored weights
                        # stored_w = []
                        # for each_syn_name in name_list_syns:
                        #     stored_w.append(br.Quantity(self.net[each_syn_name].w, copy=True))
                        self._run_network(self.stored_w, interval)
                        sd, so = self._time_train_to_bit(interval, self.architecture[num_layer + 1],
                                                        target_list[num_layer],
                                                        self.net[name_list_layers[num_layer + 1]])

                        _, P_total[:, epoch] = self._ReSuMe_optimizer(self.net[name_list_layers[num_layer]],
                                                                     self.net[name_list_layers[num_layer+1]],
                                                                     self.net[name_list_syns[num_layer]]
                                                                     , sd, so, interval, num_layer)
                        # save weights after update
                        self.stored_w = []
                        for each_syn_name in name_list_syns:
                            self.stored_w.append(br.Quantity(self.net[each_syn_name].w, copy=True))
                        # if num_layer == 1:
                        #     print(self.stored_w[1])
                        # assert False, 'Stop!!'
                        if show_output:
                            self.show_figure(Tmax, self.target[0])
                        # restore the network maybe important
                        self.net.restore()

                print('-'*10 + 'Finish weight' + str(num_layer+1) + '-'*10)
                # if num_layer == 0:
                #     print('Stop now')
                #     break
        elif method == 'simultaneously_pattern_bp':
            # target_now = self.target[0][0]  # now we just use one target in this experiment
            # target_now = [self.target[0][0], self.target[1][0]]
            target_now = []
            for i in range(self.class_num):
                for j in range(self.architecture[-1]):
                    target_now.append(self.target[i][j])
            final_target = [[e_target] for e_target in target_now]

            sub_target = []  # from left to right finally

            # If num(hidden_layer) > 1, how to program assigner ?
            for index, num_neurons in enumerate(self.architecture[1:-1][::-1]):
                # reverse it and cut the head and tails down (who does not need subsets to learn)
                pattern_list_for_one_layer = []
                for cls in range(self.class_num):
                    pattern_list_for_one_layer += [self.assigner(num_neurons, target_now)]
                sub_target.append(pattern_list_for_one_layer)
                target_now = sub_target[index]
            sub_target = sub_target[::-1]  # train from input to output
            sub_target.append(final_target)
            target_list = sub_target
            # use another train method
            # list(a.keys()) --> find all keys
            # get the name of all layers and synapses, so that we can train one by one
            name_list_layers = []
            name_list_syns = []
            for each_obj in self.net.objects:
                if 'monitor_' in each_obj.name:
                    name_list_layers.append(each_obj.name)
                elif 'syn' in each_obj.name and 'pre' not in each_obj.name:
                    name_list_syns.append(each_obj.name)
            print(name_list_syns)
            print(name_list_layers)


            # Initialize weights container
            self.stored_w = []
            for each_syn_name in name_list_syns:
                self.stored_w.append(br.Quantity(self.net[each_syn_name].w, copy=True))

            self.alpha = learning_rate  # reset the learning rate

            for epoch in range(max_epochs):
                # decay setting
                if self.decay_rate is not None:
                    if epoch % decay_epoch == 0:
                        self.alpha *= self.decay_rate
                # now I do not wanna consider multi-target task
                for cls in range(self.class_num):
                    self.net['input_sp'].set_spikes(self.input_data[cls][0], self.input_data[cls][1])
                    if self.decay_rate is not None:
                        if epoch % decay_epoch == 0:
                            self.alpha *= self.decay_rate
                    # If converged, stop it
                    if min(P_total[:, epoch - 1]) >= 0.95:
                        break
                    # Get out the stored weights
                    # stored_w = []
                    # for each_syn_name in name_list_syns:
                    #     stored_w.append(br.Quantity(self.net[each_syn_name].w, copy=True))
                    self._run_network(self.stored_w, interval)
                    for num_layer in range(len(self.architecture) - 1):

                        sd, so = self._time_train_to_bit(interval, self.architecture[num_layer + 1],
                                                        target_list[num_layer][cls],
                                                        self.net[name_list_layers[num_layer + 1]])

                        _, P = self._ReSuMe_optimizer(self.net[name_list_layers[num_layer]],
                                                                      self.net[name_list_layers[num_layer + 1]],
                                                                      self.net[name_list_syns[num_layer]]
                                                                      , sd, so, interval, num_layer)
                        # only record the performance for the final layer
                        if num_layer == len(self.architecture)-2:
                            P_total[cls, epoch] = 1 - P
                        # save weights after update
                        self.stored_w = []
                        for each_syn_name in name_list_syns:
                            self.stored_w.append(br.Quantity(self.net[each_syn_name].w, copy=True))
                        print('Finish updating layer', num_layer+1)
                        # if num_layer == 1:
                        #     print(self.stored_w[1])
                        # assert False, 'Stop!!'
                    if show_output:

                        for pattern in self.target:
                            self.show_figure(Tmax, pattern, num_neurons)
                        # restore the network maybe important
                    self.net.restore()

            for index_2, each_sw in enumerate(self.stored_w):
                print('Weights layer', index_2 + 1)
                print(each_sw)
            # show the performance curve
            # print(P_total[0, :])
            if save_weights is True:
                np.save(save_name, np.array(self.stored_w))
                print('Save Weights successfully!')
            plt.figure()
            for cls in range(self.class_num):
                plt.plot(np.linspace(1, max_epochs+1, max_epochs), P_total[cls, :], label='pattern '+ str(cls+1))
            plt.xlabel('epoch')
            plt.ylabel('error')
            plt.title('Error rate')
            plt.legend()
            plt.show()
            plt.savefig('Performance_curve.jpg')

        elif method == 'single_layer':
            # train for no
            class_num = len(self.input_data)
            for epoch in range(max_epochs):
                dwdt_total = np.zeros((self.architecture[1], self.architecture[2]))
                for cls in range(self.class_num):
                    self.net['input_sp'].set_spikes(self.input_data[cls][0], self.input_data[cls][1])
                    if self.decay_rate is not None:
                        if epoch % decay_epoch == 0:
                            self.alpha *= self.decay_rate
                    # If converged, stop it
                    if min(P_total[:, epoch - 1]) >= 0.95:
                        break

                    # Forward propagation
                    self._run_network([stored_I2H, stored_H2O], interval)
                    # what is the another not focus neuron should be?
                    sd, so = self._time_train_to_bit(interval, self.architecture[-1], self.target[cls], self.net['monitor_output'])
                    if initial == 1:  # save the initial output
                        spikeinitial_out = np.array(self.net['monitor_output'].i)
                        spikeinitial_out_t = np.array(self.net['monitor_output'].t)
                        initial = 0
                    # Backward propagation
                    dwdt, P_total[:, epoch] = self._ReSuMe_optimizer(self.net['monitor_hidden_1'], self.net['monitor_output'], self.net['syn_H2O'], sd, so, interval, 1, method='')
                    dwdt_total += dwdt
                    print('update ', epoch, ' finished')
                    stored_I2H = br.Quantity(self.net['syn_I2H'].w, copy=True)  # Thoes are constant
                    stored_H2O = br.Quantity(self.net['syn_H2O'].w, copy=True)
                    # before restore, we need this
                    if show_output:
                        self.show_figure(Tmax, self.target[cls], 1)
                    self.net.restore()

                # upadate after feed all batch
                # self.net['syn_I2H'].w[:, :] = stored_I2H
                # self.net['syn_H2O'].w[:, :] = stored_H2O
                # self.net['syn_H2O'].w[:, 0] = self.net['syn_H2O'].w[:, 0] + self.alpha * dwdt_total[:, 0]
                # stored_H2O = br.Quantity(self.net['syn_H2O'].w, copy=True)
                # stored_I2H = br.Quantity(self.net['syn_I2H'].w, copy=True)
                # print('update the weights !')

                spikefinal_out = np.array(self.net['monitor_output'].i)
                spikefinal_out_t = np.array(self.net['monitor_output'].t)
                if save_weights:
                    self.save_weights_to_mat(self.weights_container['syn_H2O'], spikeinitial_out_t, spikeinitial_out, spikefinal_out_t, spikefinal_out, P_total)

            # save weights
            self.stored_w = [stored_I2H, stored_H2O]
            br.figure()
            br.plot(self.weights_container['syn_H2O'])
            br.title('Performance Curve')
            br.xlabel('epoch')
            br.show()
            plt.ioff()
            plt.show()

    def _run_network(self, weights, interval):
        '''
        Forward propagation of the network
        :param weights: list
                weights list of all synapses
        :param interval: int
                simulation time for one epoch
        :return: None
        '''
        # Forward propagation
        self.net['syn_I2H'].w[:, :] = weights[0]
        self.net['syn_H2O'].w[:, :] = weights[1]
        self.weights_container['syn_H2O'].append(np.array(self.net['syn_H2O'].w))
        self.weights_container['syn_I2H'].append(np.array(self.net['syn_I2H'].w))
        self.net.run(interval * br.ms)

    def _ReSuMe_optimizer(self, input, output, syn_update, sd, so, interval, num_layer, method=None):
        '''

        :param input:
        :param output:
        :param syn_update:
        :param sd:
        :param so:
        :param interval:
        :param num_layer:
        :return:
        '''
        # Assume brain passes the reference of the variables
        dwdt = np.zeros((self.architecture[num_layer], self.architecture[num_layer+1]))
        learning_window = np.exp(-np.arange(0, 11)/5.0)

        for pattern in range(self.architecture[num_layer+1]):
        # for pattern in range(1):
            P = self.compute_error(output, sd[:, pattern], pattern, so[:, pattern], interval)
            # input spikes into bit stream
            # for i in range(1):
            for i in range(self.architecture[num_layer]):

                spike_hidden = input.spike_trains()
                sh = np.zeros((int(interval / self.dt) + 1, self.architecture[num_layer+1]))
                for ti in spike_hidden[i]:
                    ti = (float(ti) * 1000 / 0.1 <= interval/0.1) and int(float(ti) * 1000 / 0.1) or int(interval / 0.1)
                    sh[ti, pattern] = 1

                conv = convolve(self.parameters['A'] * sh[:, pattern], learning_window)
                # compute the sum
                dwdt[i, pattern] = sum((1.0 / self.architecture[num_layer]) * (
                        self.parameters['lr'] * (sd[:, pattern] - so[:, pattern]) + (
                        sd[:, pattern] - so[:, pattern]) * conv))

                # if num_layer == 1:
                #     print("-"*10 + 'dwdt for pattern' + str(pattern) + 'and neuron' + str(i) + "-"*10)
                #     print(dwdt)
                if method != 'update_after_compute':
                    syn_update.w[i, pattern] = syn_update.w[i, pattern] + self.alpha * dwdt[i, pattern]
            if method != 'update_after_compute':
                print('Update the weights!')
        return dwdt, P

    def compute_error(self, output, sd, pattern, so, interval):
        '''

        :param output:
        :param sd:
        :param pattern:
        :param so:
        :param interval:
        :return:
        '''
        spike_out = output.spike_trains()
        # print(spike_out[pattern])
        for ti in spike_out[pattern]:
            # clip the ti to avoid overflow
            # future will replace 0,1 to dt !!
            ti = (float(ti) * 1000 / 0.1 <= interval/0.1) and int(float(ti) * 1000 / 0.1) or int(interval/0.1)
            so[ti] = 1
            # so[int(float(ti) * 1000 / 0.1)] = 1

        if len(spike_out[pattern]) == 0:
            return 0
            # Create kernel
        g = np.sqrt(2 * 3.1415926) * 4 * Gaussian1DKernel(stddev=4)

        # Convolve data
        vd = convolve(sd, g)
        vo = convolve(so, g)
        C = np.dot(vd, vo) / (np.linalg.norm(vd) * np.linalg.norm(vo) + 1e-5)

        print('return similarity for pattern ', pattern + 1)
        print(C)

        return C

    def predict(self, input_data):
        print('-'*10 + 'Begin to predict ' + 10*'-')
        predict_label = []
        for each_input in input_data:
            self.net['input_sp'].set_spikes(each_input[0], each_input[1])
            error_list = []
            for cls in range(self.class_num):
                for pattern in range(self.architecture[-1]):
                    self._run_network(self.stored_w, self.interval)
                    sd, so = self._time_train_to_bit(self.interval, self.architecture[-1], self.target[cls],
                                                    self.net['monitor_output'])
                    error_list.append(self.compute_error(self.net['monitor_output'], sd[:, pattern], pattern, so[:, pattern], self.interval))
                    # self.net.restore()
            self.net.restore()
            predict_label.append(np.argmax(error_list))
            # print(error_list)
        return predict_label

    def predict_figure(self, input_data, target, load_model_path=None):
        print('-'*10 + 'Begin to predict ' + 10*'-')
        self.stored_w = np.load(load_model_path)
        similarity_list = []
        for each_input in input_data:
            self.net['input_sp'].set_spikes(each_input[0], each_input[1])
            for pattern in range(self.architecture[-1]):
                self._run_network(self.stored_w, self.interval)
                sd, so = self._time_train_to_bit(self.interval, self.architecture[-1], target,
                                                self.net['monitor_output'])
                similarity_list.append(self.compute_error(self.net['monitor_output'], sd[:, pattern], pattern, so[:, pattern], self.interval))
                # self.net.restore()
            self.net.restore()

            # print(error_list)
        return similarity_list

    def assigner(self, num_neurons, target, mini_sp_num=1):
        '''
        To assign the subsets automatically
        :param num_neurons: int
                the number of neurons in the previous layer(which layer needs subset)
        :param sd: list
                list of spike trains
        :return: assigned target of one layer
        '''
        if not isinstance(target[0], list):  # To normalize the format
            target = [target]
        target_num = len(target)
        all_assigned_pattern = []

        for each_target in target:
            all_assigned_pattern.append(np.array(self.pattern_decompose(each_target, num_neurons, mini_sp_num=mini_sp_num, max_sp_num=len(each_target)+1)))
        # and then we need evaluate the union of all this pattern, but now we just have one.
        # now it just support single back propagation layer
        # do sth to combine the pattern train, like union
        # now only union 2 pattern for a neuron, prevent too dense spike trains
        if target_num == 1:
            return all_assigned_pattern[0]
        processed_group = []
        for order_n in range(num_neurons):
            idx = np.argsort(np.random.random(target_num))
            processed_group.append(np.union1d(all_assigned_pattern[idx[0]][order_n], all_assigned_pattern[idx[1]][order_n]))
        #print(processed_group)
        #assert False
        #all_assigned_pattern = all_assigned_pattern[0]
        return processed_group

    def pattern_decompose(self, input_pattern, num_subset, mini_sp_num=1, max_sp_num=4):
        '''
        For pattern back propagation to decompose the input pattern
        :param input_pattern: list
                input desired pattern, a list of spike times
        :param num_subset: int
                how many subsets are needed
        :return: list of sets
                subsets of the input pattern
        '''
        if len(input_pattern) == 0:
            return [[] for _ in range(num_subset)]
        # mini_sp_num = 0
        # max_sp_num = 2

        num_samples = np.random.randint(low=min(mini_sp_num, max_sp_num-1), high=max_sp_num, size=num_subset)
        # Is it necessary to check the union of all sets equals to the target?
        return [random.sample(input_pattern, num_samples[i]) for i in range(num_subset)]

    def show_figure(self, Tmax, target, num_neuron):
        plt.clf()
        plt.ion()
        step = 0.001
        time_series = np.arange(0, Tmax, step)
        spike_trains_o = np.zeros(len(time_series))
        spike_trains_d = np.zeros(len(time_series))
        for spike_time_o in self.net['monitor_output'].t/br.ms:
            spike_time_o = (int(spike_time_o/step) < len(time_series)) and int(spike_time_o/step) or len(time_series)-1
            spike_trains_o[spike_time_o] = 1
        for spike_time_d in target[0]:
            spike_trains_d[int(spike_time_d/step)] = 1

        plt.figure(1, figsize=(8, 8))
        plt.subplot(211)
        plt.plot(time_series, spike_trains_o, label='Output spike train')
        plt.plot(time_series, spike_trains_d, label='Desired spike train')
        plt.xlabel('t/ms')
        plt.title('Output neuron ' + str(num_neuron))
        plt.legend()
        plt.ylim(-0.8, 2.2)
        plt.subplot(212)
        plt.plot(self.net['monitor_output'].t/br.ms, self.net['monitor_output'].i, 'k*', label='Output spike train')
        plt.plot(target[0], 0.2*np.ones_like(target[0]), 'r*', label='Desired spike train')
        plt.xlim(-10, 210)
        plt.ylim(-.3, 2)
        # plt.title('Output neuron ' + str(num_neuron))
        plt.pause(0.3)
        plt.show()
        #plt.close()

    def save_weights_to_mat(self, Weighttotal, spikeinitial_out_t, spikeinitial_out, spikefinal_out_t, spikefinal_out, P_total):
        sio.savemat('outputsaved',
                    {'Weighttotal': Weighttotal, 'spikeinitial_out_t': spikeinitial_out_t,
                     'spikeinitial_out': spikeinitial_out, 'spikefinal_out_t': spikefinal_out_t,
                     'spikefinal_out': spikefinal_out, 'P_total': P_total})
    def get_net(self):
        return self.net


