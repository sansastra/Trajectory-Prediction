import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # for removing unnecessary warnings
from absl import logging

logging._warn_preinit_stderr = 0
logging.warning('...')
import joblib
from sklearn.model_selection import train_test_split
from load_data_traj import load_all_data, load_all_data_ENU, load_all_data_ENU_1track, get_input_output, load_test_data
from process import scale_data, reshape_data, model_evaluate, test_track
from models import get_model




INPUT_LEN = 10 # same as timesteps
TARGET_LEN = 10
algo = 'seq' # lstm, seq, seq_at, albert
scaling = True  # False
ENU = False
track_to_check = 167

if ENU:
    features = ['x', 'y','z', 'cog', 'sog']
    #features = ['x', 'y', 'cog', 'sog']
    dim = len(features)
    input_data, target_data = load_all_data_ENU(dim, INPUT_LEN, TARGET_LEN)
    # input_data, target_data = load_all_data_ENU_1track(dim, INPUT_LEN, TARGET_LEN)
else:
    features = ['x', 'y', 'cog', 'sog']
    dim = len(features)
    input_data, target_data = load_all_data(dim, INPUT_LEN, TARGET_LEN)

data_train, data_test, target_train, target_test = train_test_split(input_data, target_data, test_size=0.30)

X_train = data_train
X_test = data_test
Y_train = target_train

if scaling:
    X_train = scale_data(data_train)
    X_test = scale_data(data_test)
    Y_train = scale_data(target_train)


if algo != 'albert':
    X_train.shape = (X_train.shape[0], INPUT_LEN, dim)
    X_test.shape = (X_test.shape[0], INPUT_LEN, dim)
    Y_train.shape = (Y_train.shape[0], INPUT_LEN, dim)

# model
model = get_model(algo, INPUT_LEN, TARGET_LEN, dim)
model.summary()
# make predictions

train_history = model.fit(X_train[:,:], Y_train[:,:], epochs=50, batch_size=1, verbose=2)
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)

# save the model to disk
if algo == 'seq':
    filename = 'Seq2Seq_model_new.h5'
elif algo == 'lstm':
    filename = 'lstm_model.h5'
elif algo == 'seq_at':
    filename = 'Seq_at_model.h5'

model.save(filename)
#joblib.dump(model, filename)

# evaluate model
model_evaluate(train_history, trainPredict, testPredict, target_train, target_test, INPUT_LEN, dim, scaling)
# test a track
# test_track(INPUT_LEN, TARGET_LEN, features, dim, track_to_check, model, ENU)


