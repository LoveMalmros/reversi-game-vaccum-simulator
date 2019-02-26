import numpy as np
from sklearn import preprocessing, linear_model
import matplotlib.pyplot as plt
import matplotlib as mpl


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


# stochastic gradient descent create weights w0 and w1: y_pred = w1 * x + w0
def stochastic_gradient_descent(x, y, error, alpha):
    # init weights
    w0 = 0.001
    w1 = 0.001

    new_weights = stochastic_sum(x[0], y[0], w0, w1, alpha)

    print('Initialized new weights: ' + str(new_weights))
    counter = 0

    for i in range(1000):
        w0 += new_weights[0]
        w1 += new_weights[1]
        #print('new w0 and w1: ' + str(w0) + '  ' + str(w1))

        new_weights = stochastic_sum(x[counter], y[counter], w0, w1, alpha)
        # w0, w1 = ... instead?

        if counter == len(x)-1:
            counter == 0
        else:
            counter += 1
    return w0, w1


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

    return new_w0, new_w1


# divide dataset into corresponding language and sort
en_x = X_train[:15, 0]
en_y = X_train[:15, 1]
norm_en_x = norm_X[:15, 0]
norm_en_y = norm_X[:15, 1]

fr_x = X_train[15:, 0]
fr_y = X_train[15:, 1]
norm_fr_x = norm_X[15:, 0]
norm_fr_y = norm_X[15:, 1]

# alpha = 0.025  # learning rate
alpha = 1e-9  # used when not normalized
error = 1e-3  # use lower value for normalized data

w0, w1 = stochastic_gradient_descent(en_x, en_y, error, alpha)

plt.figure(1)
plt.plot(fr_x, fr_y, 'bs', en_x, en_y, 'g^')
#plt.legend('French', 'English')
plt.xlabel('Number of characters')
plt.ylabel('Number of A:s')
plt.plot(en_x, w1*en_x + w0, 'r--')
plt.show()
