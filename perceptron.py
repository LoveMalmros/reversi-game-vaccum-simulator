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


# Train
classifier = linear_model.LinearRegression()
model = classifier.fit(X_train, y_train)
# print(model)

# Predict
y_test_predicted = classifier.predict(X_train)
# print(y_test_predicted)

# fetch weights
# w0 = y_test_predicted[0::2]
# w1 = y_test_predicted[1::2]


def sigmoid(x):
  return 1 / (1 + np.exp(-x))

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


def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))


def stoch_step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += (y - ((m_current * x) + b_current))
        m_gradient += x * (y - ((m_current * x) + b_current))
    new_b = b_current + (learningRate * b_gradient)
    new_m = m_current + (learningRate * m_gradient)
    return [new_b, new_m]


def batch_step_gradient(b_current, m_current, points, learningRate, i,  q):
    divider = learningRate / q
    batch_b = 0
    batch_m = 0
    for j in range(i, i+q):
        x = points[j, 0]
        y = points[j, 1]
        batch_b = (y - ((m_current * x) + b_current))
        batch_m = x * (y - ((m_current * x) + b_current))
    new_b = b_current + divider*(learningRate * batch_b)
    new_m = m_current + divider*(learningRate * batch_m)
    return [new_b, new_m]


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations, stoch):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        if(stoch):
            b, m = stoch_step_gradient(b, m, points, learning_rate)
        else:
            j = i % 5
            b, m = batch_step_gradient(b, m, points, learning_rate, j*3, 3)
    return [b, m]

def multi_gradient_descent_runner(points, ys, starting_b, starting_m1, starting_m2, learning_rate, num_iterations, stoch):
    b = starting_b
    m1 = starting_m1
    m2 = starting_m2
    for i in range(num_iterations):
        if(stoch):
            b, m1, m2 = multi_stoch_step_gradient(b, m1, m2, points, ys, learning_rate)
    return [b, m1, m2]

def multi_stoch_step_gradient(b_current, m1_current, m2_current, points, ys, learningRate):
    b_gradient = 0
    m1_gradient = 0
    m2_gradient = 0
    for i in range(0, len(points)):
        x1 = points[i, 0]
        x2 = points[i, 1]
        y = ys[i]
        b_gradient += (y - ((m1_current * x1) + (m2_current * x2) + b_current))
        m1_gradient += x1 * (y - ((m1_current * x1) + (m2_current * x2) + b_current))
        m2_gradient += x2 * (y - ((m1_current * x1) + (m2_current * x2) + b_current))
    new_b = b_current + (learningRate * b_gradient)
    new_m1 = m1_current + (learningRate * m1_gradient)
    new_m2 = m2_current + (learningRate * m2_gradient)
    return [new_b, new_m1, new_m2]

def gradient_ascent_runner(points, ys, starting_m1, starting_m2, starting_b, learning_rate, num_iterations, stoch):
    m1 = starting_m1
    m2 = starting_m1
    b = starting_m1
    for i in range(num_iterations):
        if(stoch):
            temp = []
            temp = stoch_step_ascent(m1, m2, b, points, ys, learning_rate)
            m1 = temp[0]
            m2 = temp[1]
            b = temp[2]
    return [m1, m2, b]


def linear_classifier(b, m, x1, x2):
    #print(m*x1 + b - x2)
    if m*x1 + b - x2 > 0:
        return ENGLISH_CLASSIFIER
    else:
        return FRENCH_CLASSIFIER


def reader_sparse_data(data):
    data_split = data.split(' ')
    classifier = data_split[0]
    x1 = data_split[1].split(':')[1]
    x2 = data_split[2].split(':')[1]
    return classifier, x1, x2


def leave_one_out_validation(sparse_data):
    data = sparse_data.split('\n')
    for i, line in enumerate(data):
        temp_norm_X = np.delete(norm_X, i, 0)
        [b, m] = gradient_descent_runner(temp_norm_X, 0, 0, 1e-3, 5000, True)
        [c, x1, x2] = reader_sparse_data(line)
        print(int(c) == linear_classifier(b, m, float(x1), float(x2)))


def leave_one_out_validation_logistic(sparse_data):
    data = sparse_data.split('\n')
    for i, line in enumerate(data):
        temp_norm_X = np.delete(norm_X, i, 0)
        temp_ys = np.delete(y_train, i, 0)
        #[[m1, m2],e] = gradient_ascent_runner(temp_norm_X, 0, 0, 0, 0.27, 3000, True)
        #[[m1, m2],e] = train(temp_norm_X, 0, 0, 0, 0.27, 3000, True)
        [b, m1, m2] = multi_gradient_descent_runner(temp_norm_X, temp_ys, 0, 0, 0, 0.02, 1000, True)
        plt.figure(i)
        #[[m1, m2], cost] = train(temp_norm_X, temp_ys, [0.0001, 0.0001], 0.5, 1000)
        plt.plot(norm_fr_x, norm_fr_y, 'bs', norm_en_x, norm_en_y, 'g^')
        plt.plot(norm_x, (b-(m1*norm_x))/m2, 'r')
        plt.show()
        [c, x1, x2] = reader_sparse_data(line)
        print(int(c) == logistic_classifier(m1, m2, b, float(x1), float(x2)))


# not returning correct weights, all under 0.5..?
def stoch_step_ascent(x1_current, x2_current, b_current, data, ys, learningRate):
    x1_gradient = 0
    x2_gradient = 0
    b_gradient = 0
    for i in range(0, len(data)):
        x1 = data[i, 0]
        x2 = data[i, 1]
        y = ys[i]
        x1_gradient += (y - (1 / (1 + np.exp(-(x1_current * x1)))))
        x2_gradient += (y - (1 / (1 + np.exp(-(x2_current * x2)))))
        b_gradient += (y - (1 / (1 + np.exp(-b_current))))
    new_x1 = x1_current + (learningRate * x1 * x1_gradient)
    new_x2 = x2_current + (learningRate * x2 * x2_gradient)
    new_b = b_current + (learningRate * b_gradient)
    return (new_x1, new_x2, new_b)

def logistic_classifier(m1, m2, b, x1, x2):
    # probability of being in class FRENCH (1) given x
    P = sigmoid((m1 * x1 + m2 * x2 + b))
    print(P)
    if P > 0.5:
        return FRENCH_CLASSIFIER
    else:
        return ENGLISH_CLASSIFIER

# TODO implement stop criterion as least misclassified samples, see slides
# känns som perceptronen bör felklassificera några eftersom de vill att man gör logistic regression (och det ovan)
# som enligt litteratur skulle vara mycket bättre..?
# TODO learningRate which changes as 1000/(1000 + iteration)


norm_en_x = norm_X[:15, 0]
norm_fr_x = norm_X[15:, 0]
norm_en_y = norm_X[:15, 1]
norm_fr_y = norm_X[15:, 1]
norm_x = norm_X[:, 0]
# leave_one_out_validation(parse_sparse_format(norm_X))

#[b_en, m_en] = gradient_descent_runner(norm_X[:15], 0, 0, 1e-1, 1000, True)
#[b_fr, m_fr] = gradient_descent_runner(norm_X[15:], 0, 0, 1e-1, 1000, True)

# BATCH
#[b_en2, m_en2] = gradient_descent_runner(norm_X[:15], 0, 0, 1e-1, 1000, False)
#[b_fr2, m_fr2] = gradient_descent_runner(norm_X[15:], 0, 0, 1e-1, 1000, False)

#[b, m] = gradient_descent_runner(norm_X, 0, 0, 1e-3, 5000, True)

# logistic regression
#w = gradient_ascent_runner(norm_X, 0, 1e-3, 5000, True)
#print(w)
leave_one_out_validation_logistic(parse_sparse_format(norm_X))
#[[m1, m2], cost] = train(norm_X, y_train, [0, 0], 0.1, 1000)
"""
fig = plt.figure()
ax = fig.gca(projection='3d')
x_t = np.linspace(10000, 60000, len(e))
y_t = np.linspace(500, 5000, len(e))
surf = ax.plot_trisurf(x_t, y_t, e, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()
"""
plt.figure(1)
plt.plot(norm_fr_x, norm_fr_y, 'bs', norm_en_x, norm_en_y, 'g^')
# #plt.legend('French', 'English')
# plt.xlabel('Number of characters')
# plt.ylabel('Number of A:s')
#plt.plot(norm_en_x, m_en*norm_en_x + b_en, 'g')
#plt.plot(norm_fr_x, m_fr*norm_fr_x + b_fr, 'b')

#plt.plot(norm_x, (-(m1*norm_x))/m2, 'r')
# #plt.plot(norm_x, m*norm_x + b, 'r')
# #plt.plot(fr_x, w1_fr*fr_x + w0_fr, 'b--')

plt.show()
