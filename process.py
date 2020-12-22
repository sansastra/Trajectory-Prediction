from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import math
from PythonCode.Trajectory_Prediction.load_data_traj import load_test_data, load_test_interpolated_data
import matplotlib.pyplot as plt

scalar = MinMaxScaler(feature_range=(0, 1))


# data pre-processing
def scale_data(x_train):
    return scalar.fit_transform(x_train)


def get_inverse_transform(x_train):
    return scalar.inverse_transform(x_train)


def reshape_data(x_train, INPUT_LEN, dim):
    # samples_train = x_train.shape[0]
    # x_train = x_train[:samples_train][:]
    print('size = ', x_train.shape[0])
    x_train.shape = (x_train.shape[0], INPUT_LEN, dim)
    return x_train


def model_evaluate(train_history, trainPredict, testPredict, target_train, target_test, INPUT_LEN, dim, scaling):
    plt.plot(range(len(train_history.history['loss'])), train_history.history['loss'])
    plt.savefig("loss_seq2seq.png")
    plt.pause(0.001)
    trainPredict = trainPredict.reshape(trainPredict.__len__(), INPUT_LEN * dim)

    testPredict = testPredict.reshape(testPredict.__len__(), INPUT_LEN * dim)
    if scaling:
        # invert predictions
        trainPredict = scalar.inverse_transform(trainPredict)
        testPredict = scalar.inverse_transform(testPredict)

    trainScore = math.sqrt(mean_squared_error(target_train[:, 0], trainPredict[:, 0])) + \
                 math.sqrt(mean_squared_error(target_train[:, 1], trainPredict[:, 1]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(target_test[:, 0], testPredict[:, 0])) + \
                math.sqrt(mean_squared_error(target_test[:, 1], testPredict[:, 1]))
    print('Test Score: %.2f RMSE' % (testScore))



def test_track(INPUT_LEN, TARGET_LEN, features, dim, track_to_check, model, ENU=False):
    original_data, test_data, target_data = load_test_interpolated_data(INPUT_LEN, TARGET_LEN, features, dim, track_to_check)

    # data pre-processing
    data_test = test_data
    data_test = scale_data(test_data)

    X_test = reshape_data(data_test, INPUT_LEN, dim)

    test_predict = model.predict(X_test)

    # testPredict = model.predict(X_test_broken)

    # invert predictions
    test_predict.shape = (data_test.shape[0], INPUT_LEN * dim)
    test_predict = get_inverse_transform(test_predict)

    test_predict[test_predict <= 0] = np.nan

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(original_data[:, 0], original_data[:, 1], '.k', label='Original trajectory')
    # ax.plot(original_data[-1, range(dim, INPUT_LEN*dim, dim)], original_data[-1, range(dim+1, INPUT_LEN*dim, dim)], '.m')

    ax.plot(test_predict[:, 0], test_predict[:, 1], '.b', label='Predicted trajectory')
    ax.plot(test_predict[-1, range(dim, TARGET_LEN * dim, dim)],
            test_predict[-1, range(dim + 1, TARGET_LEN * dim, dim)], '.b')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    # plt.title('AIS on-off switching anomaly detection')
    ax.legend()
    plt.show()
    plt.savefig('trajectory_pred1.pdf')
    plt.pause(0.001)