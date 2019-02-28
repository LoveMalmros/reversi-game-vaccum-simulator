import numpy as np
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


def gradient_ascent_runner(points, starting_w, learning_rate, num_iterations, stoch):
    w = starting_w
    # print(w)
    for i in range(num_iterations):
        if(stoch):
            w = stoch_step_ascent(w, points, learning_rate)
        # else:
        #    j = i % 5
        #    w = batch_step_ascent(w, points, learning_rate, j*3, 3)
    return w


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
        w = gradient_ascent_runner(temp_norm_X, 0, 1e-3, 1000, True)
        [c, x1, x2] = reader_sparse_data(line)
        print(int(c) == logistic_classifier(w, float(x1)))


# not returning correct weights, all under 0.5..?
def stoch_step_ascent(w_current, data, learningRate):
    w_gradient = 0
    for i in range(0, len(data)):
        x = data[i, 0]
        y = data[i, 1]
        w_gradient += (y - (1 / (1 + np.exp(-w_current * x))))
    new_w = w_current + (learningRate * w_gradient)
    return new_w


def logistic_classifier(w, x):
    # probability of being in class FRENCH (1) given x
    P = (1 / (1 + np.exp(-w * x)))
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
w = gradient_ascent_runner(norm_X, 0, 1e-3, 5000, True)
print(w)
leave_one_out_validation_logistic(parse_sparse_format(norm_X))

"""
fig = plt.figure()
ax = fig.gca(projection='3d')
x_t = np.linspace(10000, 60000, len(e))
y_t = np.linspace(500, 5000, len(e))
surf = ax.plot_trisurf(x_t, y_t, e, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()
"""
# plt.figure(1)
# plt.plot(norm_fr_x, norm_fr_y, 'bs', norm_en_x, norm_en_y, 'g^')
# #plt.legend('French', 'English')
# plt.xlabel('Number of characters')
# plt.ylabel('Number of A:s')
# #plt.plot(norm_en_x, m_en*norm_en_x + b_en, 'g')
# #plt.plot(norm_fr_x, m_fr*norm_fr_x + b_fr, 'b')
# #plt.plot(norm_x, m*norm_x + b, 'r')
# #plt.plot(fr_x, w1_fr*fr_x + w0_fr, 'b--')

# plt.show()
