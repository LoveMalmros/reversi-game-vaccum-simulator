import numpy as np
from sklearn import preprocessing, linear_model
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# divides English and French to two different classes: 0 & 1.
y_train = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

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
def parse_sparse_format():
    format_string = '0 '
    for x in X_train[:15]:
        format_string += '1:' + str(x[0]) + ' '
        format_string += '2:' + str(x[1]) + ' '
    format_string += '\n1 '
    for x in X_train[15:]:
        format_string += '1:' + str(x[0]) + ' '
        format_string += '2:' + str(x[1]) + ' '
    return format_string
print(parse_sparse_format())

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

# stochastic gradient descent create weights w0 and w1: y_pred = w1 * x + w0
def stochastic_gradient_descent(x, y, error, alpha):
    # init weights
    w0 = 0.001
    w1 = 0.001
    e = 0.001
    w0_list = []
    w1_list = []
    e_list = []
    new_weights = stochastic_sum(x[0], y[0], w0, w1, alpha)

    print('Initialized new weights: ' + str(new_weights))
    counter = 0

    for i in range(100):
        w0 += new_weights[0]
        w1 += new_weights[1]
        e += new_weights[2]
        w0_list.append(new_weights[0])
        w1_list.append(new_weights[1])
        e_list.append(new_weights[2])
        #print('new w0 and w1: ' + str(w0) + '  ' + str(w1))

        new_weights = stochastic_sum(x[counter], y[counter], w0, w1, alpha)
        # w0, w1 = ... instead?

        if counter == len(x)-1:
            counter == 0
        else:
            counter += 1
    return w0_list, w1_list, e_list


def stochastic_sum(x, y, w0, w1, alpha):
    new_w0 = w0  # init with 0 instead?
    new_w1 = w1

    # total squared error
    sqerror = 0

    # error for specific x, y
    error = 0

    # predicted y = f(x)
    yhat = 0

    # single squared error
    se = 0

    # update weights
    yhat = w1 * x + w0
    error = y - yhat
    se = error ** 2  # ** means 'power of'
    sqerror += se
    new_w0 = alpha * error
    new_w1 = alpha * error * x

    return new_w0, new_w1, sqerror


# divide dataset into corresponding language and sort
"""
en_x = X_train[:15, 0]
en_y = X_train[:15, 1]

fr_x = X_train[15:, 0]
fr_y = X_train[15:, 1]
# alpha = 0.025  # learning rate
alpha = 1e-5  # used when not normalized
error = 1e-10  # use lower value for normalized data

w0, w1, e = stochastic_gradient_descent(norm_en_x, norm_en_y, error, alpha)

w0_fr, w1_fr, e_fr = stochastic_gradient_descent(fr_x, fr_y, error, alpha)
"""


norm_en_x = norm_X[:15, 0]
norm_en_y = norm_X[:15, 1]
norm_fr_x = norm_X[15:, 0]
norm_fr_y = norm_X[15:, 1]
[b_en, m_en] = gradient_descent_runner(norm_X[:15], 0, 0, 1e-1, 1000, True)
[b_fr, m_fr] = gradient_descent_runner(norm_X[15:], 0, 0, 1e-1, 1000, True)
[b_en2, m_en2] = gradient_descent_runner(norm_X[:15], 0, 0, 1e-1, 1000, False)
[b_fr2, m_fr2] = gradient_descent_runner(norm_X[15:], 0, 0, 1e-1, 1000, False)
print(b_en)
print(m_en)
"""
fig = plt.figure()
ax = fig.gca(projection='3d')
x_t = np.linspace(10000, 60000, len(e))
y_t = np.linspace(500, 5000, len(e))
surf = ax.plot_trisurf(x_t, y_t, e, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()"""
plt.figure(1)
plt.plot(norm_fr_x, norm_fr_y, 'bs', norm_en_x, norm_en_y, 'g^')
#plt.legend('French', 'English')
plt.xlabel('Number of characters')
plt.ylabel('Number of A:s')
plt.plot(norm_en_x, m_en*norm_en_x + b_en, 'g--')
plt.plot(norm_fr_x, m_fr*norm_fr_x + b_fr, 'b--')
#plt.plot(fr_x, w1_fr*fr_x + w0_fr, 'b--')
plt.show()
