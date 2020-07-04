import numpy as np


BATCHES = [5, 50, 100]
ITERS = 150


def get_subgradient(w, reg_param, data, labels):
    sample = data[0]
    sample_length = len(sample)
    m = len(data)
    only_w = w[: len(w) - 1]
    b = w[-1]
    sm = np.zeros(sample_length+1)
    for i in range(m):
        y_i = labels[i]
        x_i = data[i]
        extended_x_i = np.append(data[i], 1)
        curr = y_i * extended_x_i
        if y_i * (only_w @ x_i + b) < 1:
            sm += curr
    extended_w = np.append(only_w, 0)
    return reg_param * extended_w - 1/m * sm



def gd(data, labels, iters, eta, w, reg_param):
    """
    The sub gradient descent algorithm
    :param data: An n by d numpy array, where n is the amount of samples and
    d is the dimension of each sample
    :param labels: An n by 1 numpy array with the labels of each sample
    :param iters: An integer that will define the amount of iterations
    :param eta: A positive number that will define the learning rate
    :param w: A d+1 by 1 numpy array where d is the dimension of each sample
    i.e. the parameters of the classifier where the last element is the bias
    :param reg_param: The regularization parameter in the sub-gradient
    :return: A d+1 by 1 numpy array which contains the output of the sub
    gradient descent algorithm over "iters" iterations
    """
    for iter in range(iters):
        w += -eta * get_subgradient(w, reg_param, data, labels)
    return w


def sgd(data, labels, iters, eta, w, batch, reg_param):
    """
    The stochastic gradient descent algorithm
    :param data: An n by d numpy array, where n is the amount of samples and
    d is the dimension of each sample
    :param labels: An n by 1 numpy array with the labels of each sample
    :param iters: An integer that will define the amount of iterations
    :param eta: A positive number that will define the learning rate
    :param w: A d+1 by 1 numpy array where d is the dimension of each sample
    i.e. the parameters of the classifier where the last element is the bias
    :param batch: The amount of samples that the algorithm would draw and use at each iteration
    :param reg_param: The regularization parameter in the sub-gradient
    :return: A d+1 by 1 numpy array which contains the output of the
    sub-gradient descent algorithm over "iters" iterations
    """
    lst = []
    for iter in range(iters):
        samples_indices = np.random.choice(range(len(data)), batch)
        z_t_index = np.random.choice(samples_indices)
        z_t = data[z_t_index]
        y_t = labels[z_t_index]
        z_t_extended = np.append(z_t, 1)
        v_t = -y_t * z_t_extended
        w -= eta * v_t
        lst.append(w)
    sm = np.zeros(len(w))
    for vec in lst:
        sm += vec
    return 1/iters * sm


def get_hypothesis(w):
    def h_w(x):
        return np.sign(w[:-1] @ x + w[-1])
    return h_w


def test_error(w, test_data, test_labels):
    """
    Returns a scalar with the respective 0-1 loss for the hypothesis
    :param w: a d+1 by 1 numpy array where h_w(x) = sign(<w[:-1], x> + w[-1])
    :param test_data: An n by d numpy array with n samples
    :param test_labels: An n by 1 numpy array with the labels of the samples
    :return: A scalar with the respective 0-1 loss for the hypothesis
    """
    h_w = get_hypothesis(w)
    test_predictions = []
    for sample in test_data:
        test_predictions.append(h_w(sample))
    result = test_labels - test_data
    samples_amount = len(result)
    non_zeros_amount = np.count_nonzero(result)
    zeros_amount = samples_amount - non_zeros_amount
    return zeros_amount / samples_amount


image_size = 28
no_of_different_labels = 2
image_pixels = image_size * image_size
data_path = r'C:\Users\amitb\PycharmProjects\IML\ex6\dataset\mldata'
train_data = np.loadtxt(data_path + r"\mnist_train.csv",
                        delimiter=",")
test_data = np.loadtxt(data_path + r"\mnist_test.csv",
                       delimiter=",")

fac = 0.99 / 255
all_train_imgs = np.asfarray(train_data[:, 1:]) * fac + 0.01
all_test_imgs = np.asfarray(test_data[:, 1:]) * fac + 0.01

all_train_labels = np.asfarray(train_data[:, :1])
all_test_labels = np.asfarray(test_data[:, :1])

binary_train_indices = np.logical_or((all_train_labels == 0),
                                    (all_train_labels == 1))
binary_test_indices = np.logical_or((all_test_labels == 0),
                                    (all_test_labels == 1))

binary_train_indices = binary_train_indices.reshape(
     binary_train_indices.shape[0])
binary_test_indices = binary_test_indices.reshape(
    binary_test_indices.shape[0])

x_train, y_train = all_train_imgs[binary_train_indices], \
                   all_train_labels[binary_train_indices]
x_test, y_test = all_test_imgs[binary_test_indices], \
                 all_test_labels[binary_test_indices]

y_train = y_train.reshape((len(y_train),))
y_test = y_test.reshape((len(y_test),))
y_train[y_train == 0] = -1
y_test[y_test == 0] = -1


sample = x_train[0]
w_length = len(sample) + 1
zero_w = np.zeros(w_length)
random_w = np.random.rand(w_length)
eta = np.random.rand(1)[0]
reg_param = np.random.randint(0, 10, 1)[0] + np.random.rand(1)[0]


gd(x_train, y_train, ITERS, eta, random_w, reg_param)
