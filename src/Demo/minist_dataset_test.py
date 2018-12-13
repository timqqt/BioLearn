import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from brian2 import *
import random

digits = datasets.load_digits()
NData = len(digits.images)
data = {'x': digits.images.reshape((NData, -1)), 'y': digits.target}

testFraction = 0.6
T, S = {}, {}
T['x'], S['x'], T['y'], S['y'] = train_test_split(data['x'], data['y'],
                                                  test_size=testFraction, random_state=0)

NT, NS = len(T['y']), len(S['y'])
print(T['y'])

def evaluate(h, T, S, name):
    def errorRate(h, S):
        x, y = S['x'], S['y']
        return (1 - h.score(x, y)) * 100

    f = '{:s}: training error rate is {:.2f}, test error rate is {:.2f}'
    err = (errorRate(h, T), errorRate(h, S), name)
    print(f.format(name, err[0], err[1]))
    return err

def img2spike(img, threshold=8, coe=4):
    img = [0] * 10 + list(img) + [0] *10
    img = np.array(img)
    img = np.where(img>=threshold, 1, 0)
    spike_train = np.where(img == 1)
    return spike_train[0]*coe

def generate_input_data(spt):
    num_samples = np.random.randint(low=2, high=10, size=20)
    return [random.sample(list(spt), num_samples[i]) for i in range(20)]

def generate_brian_data(input_spike_times):
    input_indices = np.empty(0)
    input_spike_train = np.empty(0)
    for index, each_i in enumerate(input_spike_times):
        i_indices = np.repeat(index, len(each_i), axis=0)
        input_indices = np.concatenate((input_indices, i_indices), axis=0, out=None)
        input_spike_train = np.concatenate((input_spike_train, np.array(each_i)), axis=0, out=None)
    return input_indices, input_spike_train*ms
digit_1 = {'x': T['x'][5], 'y': T['y'][5]}
digit_2 = {'x': T['x'][8], 'y': T['y'][8]}
plt.imshow(np.array(T['x'][15]).reshape(8, 8))
plt.show()
plt.imshow(np.array(T['x'][10]).reshape(8, 8))
plt.show()
# padding to 64 -> 100 * ms work
spike_train1 = img2spike(digit_1['x'], 8)
spike_train2 = img2spike(digit_2['x'], 8)
input_indices_0, input_spike_times_0 = generate_brian_data([spike_train1])
input_indices_9, input_spike_times_9 = generate_brian_data([spike_train2])
spike_train1 = [list(spike_train1)]
# plt.imshow(np.array(digit_1['x']).reshape(8, 8), cmap='gray')
# plt.show()
# plt.imshow(np.array(digit_2['x']).reshape(8, 8), cmap='gray')
# plt.show()

# generate bunch of zero
digit_0_label = {'x': T['x'][np.where(T['y']==0)], 'y': T['y'][np.where(T['y']==0)]}
digit_9_label = {'x': T['x'][np.where(T['y']==9)], 'y': T['y'][np.where(T['y']==9)]}

bunch_of_zero = []

for i in range(len(digit_0_label['y'])):
    input_indices_0_piece, input_spike_times_0_piece = generate_brian_data([img2spike(digit_0_label['x'][i], 8)])
    bunch_of_zero.append([input_indices_0_piece, input_spike_times_0_piece])

bunch_of_nine = []

for i in range(len(digit_9_label['y'])):
    input_indices_9_piece, input_spike_times_9_piece = generate_brian_data([img2spike(digit_9_label['x'][i], 8)])
    bunch_of_nine.append([input_indices_9_piece, input_spike_times_9_piece])

