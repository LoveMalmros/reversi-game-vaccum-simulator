import numpy as np
import math
from sklearn import preprocessing, linear_model
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


ENGLISH_CLASSIFIER = 0
FRENCH_CLASSIFIER = 1

# divides English and French to two different classes: 0 & 1.
y_train = np.array(
    [ENGLISH_CLASSIFIER, ENGLISH_CLASSIFIER, ENGLISH_CLASSIFIER, ENGLISH_CLASSIFIER, ENGLISH_CLASSIFIER, ENGLISH_CLASSIFIER, ENGLISH_CLASSIFIER, ENGLISH_CLASSIFIER, ENGLISH_CLASSIFIER, ENGLISH_CLASSIFIER, ENGLISH_CLASSIFIER, ENGLISH_CLASSIFIER, ENGLISH_CLASSIFIER, ENGLISH_CLASSIFIER, ENGLISH_CLASSIFIER,
     FRENCH_CLASSIFIER, FRENCH_CLASSIFIER, FRENCH_CLASSIFIER, FRENCH_CLASSIFIER, FRENCH_CLASSIFIER, FRENCH_CLASSIFIER, FRENCH_CLASSIFIER, FRENCH_CLASSIFIER, FRENCH_CLASSIFIER, FRENCH_CLASSIFIER, FRENCH_CLASSIFIER, FRENCH_CLASSIFIER, FRENCH_CLASSIFIER, FRENCH_CLASSIFIER, FRENCH_CLASSIFIER])

X_train = np.array(
    [[35680, 2217], [42514, 2761], [15162, 990], [35298, 2274],
     [29800, 1865], [40255, 2606], [74532, 4805], [37464, 2396],
     [31030, 1993], [24843, 1627], [36172, 2375], [39552, 2560],
     [72545, 4597], [75352, 4871], [18031, 1119], [36961, 2503],
     [43621, 2992], [15694, 1042], [36231, 2487], [29945, 2014],
     [40588, 2805], [75255, 5062], [37709, 2643], [30899, 2126],
     [25486, 1784], [37497, 2641], [40398, 2766], [74105, 5047],
     [76725, 5312], [18317, 1215]
     ])

# Normalize
min_max_scaler = preprocessing.MinMaxScaler()
norm_X = min_max_scaler.fit_transform(X_train)

def get_language(c):
    if c == 0:
        return 'ENGLISH'
    else:
        return 'FRENCH'

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def parse_sparse_format(x_data):
    format_string = ''
    for x in x_data[:15]:
        format_string += '0 '
        format_string += '1:' + str(x[0]) + ' '
        format_string += '2:' + str(x[1]) + '\n'
    for x in x_data[15:]:
        format_string += '1 '
        format_string += '1:' + str(x[0]) + ' '
        format_string += '2:' + str(x[1]) + '\n'
    return format_string[:-1]

def reader_sparse_data(data):
    data_split = data.split(' ')
    classifier = data_split[0]
    x1 = data_split[1].split(':')[1]
    x2 = data_split[2].split(':')[1]
    return classifier, x1, x2

def get_learning_rate(i):
    return 1000/(1000+1)

def stoch_step_gradient(b_current, m_current, points):
    learning_rate = 0.02
    b_gradient = 0
    m_gradient = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += (y - ((m_current * x) + b_current))
        m_gradient += x * (y - ((m_current * x) + b_current))
    new_b = b_current + (learning_rate * b_gradient)
    new_m = m_current + (learning_rate * m_gradient)
    return [new_b, new_m]


def batch_step_gradient(b_current, m_current, points, index, i,  q):
    learning_rate = get_learning_rate(index)
    divider = learning_rate / q
    batch_b = 0
    batch_m = 0
    for j in range(i, i+q):
        x = points[j, 0]
        y = points[j, 1]
        batch_b = (y - ((m_current * x) + b_current))
        batch_m = x * (y - ((m_current * x) + b_current))
    new_b = b_current + divider*(learning_rate * batch_b)
    new_m = m_current + divider*(learning_rate * batch_m)
    return [new_b, new_m]

def gradient_descent_logistic(X, y, num_iter):
    theta = np.zeros(X.shape[1])
    for i in range(num_iter):
        learning_rate = get_learning_rate(i)
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, (h - y)) / y.size
        theta -= learning_rate * gradient
    return theta

def gradient_descent_runner(points, starting_b, starting_m, num_iterations, stoch):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        if(stoch):
            b, m = stoch_step_gradient(b, m, points)
        else:
            j = i % 5
            b, m = batch_step_gradient(b, m, points, i, j*3, 3)
    return [b, m]


def logistic_classifier(X, theta):
    P = sigmoid(np.dot(X, theta))
    if P > 0.5:
        return FRENCH_CLASSIFIER
    else:
        return ENGLISH_CLASSIFIER

def linear_classifier(b, m, x1, x2):
    if m*x1 + b - x2 > 0:
        return ENGLISH_CLASSIFIER
    else:
        return FRENCH_CLASSIFIER


def leave_one_out_validation_linear(sparse_data, stoch, verbose):
    data = sparse_data.split('\n')
    print('Fitting and evaluating')
    number_of_true = 0
    for i, line in enumerate(data):
        temp_norm_X = np.delete(norm_X, i, 0)
        [b, m] = gradient_descent_runner(temp_norm_X, 0, 0, 1000, stoch)
        [c, x1, x2] = reader_sparse_data(line)
        pred = linear_classifier(b, m, float(x1), float(x2))
        if int(c) == pred:
            number_of_true += 1
        if verbose:
            verbose_validation(c,i ,pred)
    print('Validation finished with a prediction of ' + str(number_of_true/len(data)) + '.')
    print('Guessed ' + str(number_of_true) + ' right, out of ' + str(len(data)))

def leave_one_out_validation_logistic(sparse_data, verbose):
    data = sparse_data.split('\n')
    print('Fitting and evaluating')
    number_of_true = 0
    for i, line in enumerate(data):
        temp_norm_X = np.delete(norm_X, i, 0)
        temp_ys = np.delete(y_train, i, 0)
        theta = gradient_descent_logistic(temp_norm_X, temp_ys, 50000)
        [c, x1, x2] = reader_sparse_data(line)
        pred = logistic_classifier([float(x1), float(x2)], theta)
        if int(c) == pred:
            number_of_true += 1
        if verbose:
            verbose_validation(c, i,pred)
    print('Validation finished with a prediction of ' + str(number_of_true/len(data)) + '.')
    print('Guessed ' + str(number_of_true) + ' right, out of ' + str(len(data)))

def verbose_validation(c, index, pred):
    x = X_train[index]
    if (int(c) == pred):
        print('CORRECT PREDICTION')
    else:
        print('FALSE PREDICTION')
    print("NMBR OF WORDS: " + str(x[0]))
    print("NMBR OF A's: " + str(x[1]))
    print("LANGUAGE: " + get_language(int(c)))
    print("PREDICTED LANGUAGE: " + get_language(pred))
    print('\n')

def menu():
    print('What would you like to do?')
    print('1. Visualize stuff')
    print('2. Validate stuff')
    choice = input()
    if(int(choice) == 1):
        print('What would you like to visualize?')
        print('1. Fitted linear regression with stochastic gradient descent')
        print('2. Fitted linear regression with batch gradient descent')
        print('3. Fitted logistic regression with stochastic gradient descent')
        choice2 = input()
        if(int(choice2) == 1):
            plt.figure()
            [b, m] = gradient_descent_runner(norm_X, 0, 0, 1000, True)
            [b_en, m_en] = gradient_descent_runner(norm_X[:15], 0, 0, 1000, True)
            [b_fr, m_fr] = gradient_descent_runner(norm_X[15:], 0, 0, 1000, True)
            plt.plot(norm_fr_x, norm_fr_y, 'bs', norm_en_x, norm_en_y, 'g^')
            plt.plot(norm_x, m*norm_x + b, 'r--')
            plt.plot(norm_x, m_en*norm_x + b_en, 'g--')
            plt.plot(norm_x, m_fr*norm_x + b_fr, 'b--')
            plt.show()
        if(int(choice2) == 2):
            plt.figure()
            [b, m] = gradient_descent_runner(norm_X, 0, 0, 1000, False)
            [b_en, m_en] = gradient_descent_runner(norm_X[:15], 0, 0, 1000, False)
            [b_fr, m_fr] = gradient_descent_runner(norm_X[15:], 0, 0, 1000, False)
            plt.plot(norm_fr_x, norm_fr_y, 'bs', norm_en_x, norm_en_y, 'g^')
            plt.plot(norm_x, m*norm_x + b, 'g--')
            plt.plot(norm_x, m_en*norm_x + b_en, 'g--')
            plt.plot(norm_x, m_fr*norm_x + b_fr, 'b--')
            plt.show()
        if(int(choice2) == 3):
            plt.figure()
            [m1, m2] = gradient_descent_logistic(norm_X, y_train, 50000)
            plt.plot(norm_fr_x, norm_fr_y, 'bs', norm_en_x, norm_en_y, 'g^')
            plt.plot(norm_x, -m1*norm_x / m2, 'r--')
            plt.show()
        menu()
    elif(int(choice) == 2):
        print('Would you like to have the verbose setting on?')
        print('1. Yes')
        print('2. No')
        verbose = False
        choice_verbose = input()
        if(int(choice_verbose) == 1):
            verbose = True
        print('What would you like to validate?')
        print('1. Linear regression with stochastic gradient descent')
        print('2. Linear regression with batch gradient descent')
        print('3. Logistic regression with stochastic gradient descent')
        choice_validate = input()
        if(int(choice_validate) == 1):
            leave_one_out_validation_linear(parse_sparse_format(norm_X), True, verbose)
        if(int(choice_validate) == 2):
            leave_one_out_validation_linear(parse_sparse_format(norm_X), False, verbose)
        if(int(choice_validate) == 3):
            leave_one_out_validation_logistic(parse_sparse_format(norm_X), verbose)
        menu()
    else:
        print('try again')
        menu()


norm_en_x = norm_X[:15, 0]
norm_fr_x = norm_X[15:, 0]
norm_en_y = norm_X[:15, 1]
norm_fr_y = norm_X[15:, 1]
norm_x = norm_X[:, 0]

#START PROGRAM
menu()
