from pathlib import Path
import numpy as np
import pandas as pd
from random import randint
from numpy import array
from numpy import argmax
from math import sin, cos, sqrt, atan2, radians
#from geopy import distance
import matplotlib.pyplot as plt
import os
import h5py
import pickle
from PIL import Image
from datetime import date, time, datetime
from PythonCode.KF.ENUtransform import WGS84toENU, ENUtoWGS84
from itertools import groupby
from operator import itemgetter
import math
from os.path import join
interactive = True
headers=['index1', 'ship_nr','id','repeat_indicator','mmsi','nav_status','rot_over_range','rot','sog','position_accuracy','x','y','cog','true_heading',
         'timestamp','special_manoeuvre','spare','raim','sync_state','slot_timeout','slot_offset', 'abs_time', 'date', 'time']



SAMPLING_TIME = 30 # actually 60 seconds, because sampling on old data is already done at 2 seconds interval
legecy_seq_len = 60  # number of data per track does not include extra 60 samples, coming from legacy
nr_of_actual_vessels = 120 #30
ROSTOCK = (12.114733, 54.145409)

def load_all_data(dim, INPUT_LEN, PRED_LEN):

    ############## generate data from pickle to train ANN ##

    np.random.seed(10)
    path = Path('/home/sing_sd/Desktop/anomaly_detection/PythonCode/Trajectory_Prediction/')

    in_clm_len = dim * INPUT_LEN
    out_clm_len = dim * PRED_LEN

    filename = path / 'all_sampled_tracks_interpol_1minute.csv'
    with open(filename, 'rb') as f:
        data = pd.read_csv(f)
    data = np.array(data)



    filename = path / 'data_len_sampled_tracks_interpol_1minute.pkl'
    with open(filename, 'rb') as f:
        data_all_tracks = pickle.load(f)
    overall_data = np.full(shape=(int(sum(data_all_tracks[:nr_of_actual_vessels]) + (legecy_seq_len - 1) * nr_of_actual_vessels), in_clm_len), fill_value=np.nan)
    # make output column same as input column with appending zeros to out_column_len
    target_data = np.full(shape=(int(sum(data_all_tracks[:nr_of_actual_vessels]) + (legecy_seq_len - 1) * nr_of_actual_vessels), in_clm_len), fill_value=np.nan)

    # target_data = np.full(shape=(int(sum(data_all_tracks[:nr_of_actual_vessels])+(timesteps-1)*nr_of_actual_vessels), out_clm_len), fill_value=np.nan)
    nr_track = 0
    startIndex = 0
    for track_nr in range(nr_of_actual_vessels): #total number of vessels between 0 and 150, some vessels have less data so ignored
        if data_all_tracks[track_nr] < 500:
            continue
        nr_track += 1
        nr_sampled_data = data_all_tracks[track_nr]
        # shift from top and put on remaining columns


        # plot_trajectory(data[startIndex: startIndex + nr_sampled_data, 0:2])

        overall_data[startIndex: startIndex + nr_sampled_data, 0:dim] = data[startIndex: startIndex + nr_sampled_data, 0:dim]


        for clm_nr in range(1, INPUT_LEN):
            overall_data[startIndex: startIndex + nr_sampled_data - clm_nr,
            clm_nr * dim:(clm_nr + 1) * dim] = overall_data[
                                               startIndex + 1: startIndex + nr_sampled_data - clm_nr + 1,
                                               (clm_nr - 1) * dim:clm_nr * dim]

        target_data[startIndex:startIndex + nr_sampled_data-INPUT_LEN-PRED_LEN+1, 0:out_clm_len] = \
            overall_data[startIndex + INPUT_LEN:startIndex + nr_sampled_data - PRED_LEN+1, 0:out_clm_len]


        # clear last few rows that does not have sufficient data to make ...
        overall_data[startIndex + nr_sampled_data - INPUT_LEN-PRED_LEN+1 : startIndex + nr_sampled_data] = np.nan
        startIndex += nr_sampled_data

    overall_data = overall_data[~np.isnan(overall_data[:, 0])]
    target_data = target_data[~np.isnan(target_data[:, 0])]
    target_data = np.nan_to_num(target_data)
    print('Overall data size= ', len(overall_data))
    print("nr of tracks for training = ", nr_track)
    return overall_data, target_data

    # plt.rcParams.update({'font.size': 16})
    # fig, ax = plt.subplots(figsize=(8, 7))
    # im = Image.open('world_11_54_12-4_55.PNG')  # in degrees and minutes
    # ax.imshow(im, extent=(11, 12.6666, 54.0, 55), aspect='auto')
    # ax.plot(overall_data[:,0], overall_data[:,1], 'w.',label='Trajectories of vessels')
    # boundary_x, boundary_y = get_po_boundary()
    # ax.plot(boundary_x, boundary_y, 'k-', markersize=8, label='AIS transmission reach')
    # ax.plot(ROSTOCK[0], ROSTOCK[1], 'ko', markersize=12, label='Rostock location')
    # plt.pause(0.01)
    # ax.plot(overall_data[ind_ano[ind_po], 0], overall_data[ind_ano[ind_po], 1], 'bo', markersize=10,label='Power outage')
    # ax.plot(overall_data[ind_ano[ind_ano1], 0], overall_data[ind_ano[ind_ano1], 1], 'rx', markersize=10,label='AIS anomaly')
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.title('AIS on-off switching anomaly detection')
    # ax.legend()
    # plt.show()
    # plt.pause(0.01)

def load_all_data_ENU(dim, INPUT_LEN, PRED_LEN):

    ############## generate data from pickle to train ANN ##

    np.random.seed(10)
    path = Path('/home/sing_sd/Desktop/anomaly_detection/PythonCode/Trajectory_Prediction/')

    in_clm_len = dim * INPUT_LEN
    out_clm_len = dim * PRED_LEN

    filename = path / 'all_sampled_tracks_interpol_1minute.csv'
    with open(filename, 'rb') as f:
        data = pd.read_csv(f)
    data = np.array(data)

    # # convert lon, lat in ENU
    lon = np.transpose(data[:, 0])
    lat = np.transpose(data[:, 1])
    tREF = {"lon": 12.114733,
            "lat": 54.145409,
            "ECEF": np.array([[3660725], [785776], [514624]])
            }

    zENU = WGS84toENU(lon, lat, tREF, h=0.)

    data[:, 0] = np.transpose(zENU[0, :])
    data[:, 1] = np.transpose(zENU[1, :])
    # if no z dimension then comment next line
    data = np.insert(data, 2, np.transpose(zENU[2, :]), axis=1)  # insert zeros on 3rd column in data for

    filename = path / 'data_len_sampled_tracks_interpol_1minute.pkl'
    with open(filename, 'rb') as f:
        data_all_tracks = pickle.load(f)
    # plt.plot(data[data_all_tracks[0]:data_all_tracks[0]+data_all_tracks[1], 0], data[data_all_tracks[0]:data_all_tracks[0]+data_all_tracks[1], 1])
    # plt.show()
    overall_data = np.full(shape=(int(sum(data_all_tracks[:nr_of_actual_vessels]) + (legecy_seq_len - 1) * nr_of_actual_vessels), in_clm_len), fill_value=np.nan)
    # make output column same as input column with appending zeros to out_column_len
    target_data = np.full(shape=(int(sum(data_all_tracks[:nr_of_actual_vessels]) + (legecy_seq_len - 1) * nr_of_actual_vessels), in_clm_len), fill_value=np.nan)

    # target_data = np.full(shape=(int(sum(data_all_tracks[:nr_of_actual_vessels])+(timesteps-1)*nr_of_actual_vessels), out_clm_len), fill_value=np.nan)

    startIndex = 0
    for track_nr in range(nr_of_actual_vessels): #total number of vessels between 0 and 150, some vessels have less data so ignored
        nr_sampled_data = data_all_tracks[track_nr]
        # shift from top and put on remaining columns


        #plot_trajectory(data[startIndex: startIndex + nr_sampled_data, 0:2])

        overall_data[startIndex: startIndex + nr_sampled_data, 0:dim] = data[startIndex: startIndex + nr_sampled_data, 0:dim]


        for clm_nr in range(1, INPUT_LEN):
            overall_data[startIndex: startIndex + nr_sampled_data - clm_nr,
            clm_nr * dim:(clm_nr + 1) * dim] = overall_data[
                                               startIndex + 1: startIndex + nr_sampled_data - clm_nr + 1,
                                               (clm_nr - 1) * dim:clm_nr * dim]

        target_data[startIndex:startIndex + nr_sampled_data-INPUT_LEN-PRED_LEN+1, 0:out_clm_len] = \
            overall_data[startIndex + INPUT_LEN:startIndex + nr_sampled_data - PRED_LEN+1, 0:out_clm_len]


        # clear last few rows that does not have sufficient data to make ...
        overall_data[startIndex + nr_sampled_data - INPUT_LEN-PRED_LEN+1 : startIndex + nr_sampled_data] = np.nan
        startIndex += nr_sampled_data

    overall_data = overall_data[~np.isnan(overall_data[:, 0])]
    target_data = target_data[~np.isnan(target_data[:, 0])]
    target_data = np.nan_to_num(target_data)
    print('Overall data size= ', len(overall_data))
    return overall_data, target_data

def load_all_data_ENU_1track(dim, INPUT_LEN, PRED_LEN):

    ############## generate data from pickle to train ANN ##

    np.random.seed(10)
    path = Path('/home/sing_sd/Desktop/anomaly_detection/PythonCode/Trajectory_Prediction/')

    in_clm_len = dim * INPUT_LEN
    out_clm_len = dim * PRED_LEN

    filename = path / 'Track167_interpolated_1min.csv'
    with open(filename, 'rb') as f:
        data = pd.read_csv(f)
    data = np.array(data)

    # # convert lon, lat in ENU
    lon = np.transpose(data[:, 0])
    lat = np.transpose(data[:, 1])
    tREF = {"lon": 12.114733,
            "lat": 54.145409,
            "ECEF": np.array([[3660725], [785776], [514624]])
            }

    zENU = WGS84toENU(lon, lat, tREF, h=0.)




    data_all_tracks = len(zENU[0,:])*np.ones(shape=nr_of_actual_vessels, dtype=int)
    # plt.plot(data[data_all_tracks[0]:data_all_tracks[0]+data_all_tracks[1], 0], data[data_all_tracks[0]:data_all_tracks[0]+data_all_tracks[1], 1])
    # plt.show()
    overall_data = np.full(shape=(int(sum(data_all_tracks[:nr_of_actual_vessels]) + (legecy_seq_len - 1) * nr_of_actual_vessels), in_clm_len), fill_value=np.nan)
    # make output column same as input column with appending zeros to out_column_len
    target_data = np.full(shape=(int(sum(data_all_tracks[:nr_of_actual_vessels]) + (legecy_seq_len - 1) * nr_of_actual_vessels), in_clm_len), fill_value=np.nan)

    # target_data = np.full(shape=(int(sum(data_all_tracks[:nr_of_actual_vessels])+(timesteps-1)*nr_of_actual_vessels), out_clm_len), fill_value=np.nan)
    plt.plot(zENU[0, :], zENU[1, :], 'b.', markersize=2)
    startIndex = 0
    for track_nr in range(nr_of_actual_vessels): #total number of vessels between 0 and 150, some vessels have less data so ignored
        nr_sampled_data = data_all_tracks[track_nr]
        # shift from top and put on remaining columns
        np.random.seed(track_nr)
        uncertainty = np.random.uniform(-2000, 2000, nr_sampled_data)
        data[:, 0] = np.transpose(zENU[0, :] + uncertainty/2)
        data[:, 1] = np.transpose(zENU[1, :] + uncertainty/3)
        # if no z dimension then comment next line
        data = np.insert(data, 2, np.transpose(zENU[2, :]), axis=1)  # insert zeros on 3rd column in data for
        plt.plot(data[:, 0], data[:, 1], 'k.', markersize=2)
        plt.pause(0.001)
        #plot_trajectory(data[startIndex: startIndex + nr_sampled_data, 0:2])

        #continue
        overall_data[startIndex: startIndex + nr_sampled_data, 0:dim] = data[0: nr_sampled_data, 0:dim]


        for clm_nr in range(1, INPUT_LEN):
            overall_data[startIndex: startIndex + nr_sampled_data - clm_nr,
            clm_nr * dim:(clm_nr + 1) * dim] = overall_data[
                                               startIndex + 1: startIndex + nr_sampled_data - clm_nr + 1,
                                               (clm_nr - 1) * dim:clm_nr * dim]

        target_data[startIndex:startIndex + nr_sampled_data-INPUT_LEN-PRED_LEN+1, 0:out_clm_len] = \
            overall_data[startIndex + INPUT_LEN:startIndex + nr_sampled_data - PRED_LEN+1, 0:out_clm_len]


        # clear last few rows that does not have sufficient data to make ...
        overall_data[startIndex + nr_sampled_data - INPUT_LEN-PRED_LEN+1 : startIndex + nr_sampled_data] = np.nan
        startIndex += nr_sampled_data

    overall_data = overall_data[~np.isnan(overall_data[:, 0])]
    target_data = target_data[~np.isnan(target_data[:, 0])]
    target_data = np.nan_to_num(target_data)
    print('Overall data size= ', len(overall_data))
    return overall_data, target_data

def plot_trajectory(data):

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(8, 7))
    # im = Image.open('\Sandeep\Python\AnomalyDetection\src\world_11_54_12-4_55.PNG')  # in degrees and minutes
    # ax.imshow(im, extent=(11, 12.6666, 54.0, 55), aspect='auto')
    ax.plot(data[:,0], data[:,1], 'k.',markersize=2, label='Trajectories of vessels')
    #boundary_x, boundary_y = get_po_boundary()
    #ax.plot(boundary_x, boundary_y, 'k-', markersize=8, label='AIS transmission reach')
    #ax.plot(ROSTOCK[0], ROSTOCK[1], 'ko', markersize=12, label='Rostock location')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    # plt.title('AIS on-off switching anomaly detection')
    ax.legend()
    #plt.show()
    plt.pause(0.01)


def load_test_data(INPUT_LEN, TARGET_LEN, features,  dim, track_to_check):

    in_clm_len = INPUT_LEN*dim
    out_clm_len = TARGET_LEN * dim
    interpolate_interval = 60 # seconds
    SAMPLING_TIME = 1 # second
    path = '/home/sing_sd/Desktop/anomaly_detection/PythonCode/Resources/track_pickle/'

    filename = 'track{}'.format(track_to_check)

    try:
        data = pd.read_pickle(path + filename + '.pkl')
    except IOError:
        print("Error: File does not appear to exist for track ", track_to_check)
        return 0, 0
    # without interpolation
    original_data = np.array([data.x, data.y]).transpose()

    start_time = datetime.strptime(data.iloc[0]['date'] + ' ' + data.iloc[0]['time'],
                                   '%m/%d/%Y %H:%M:%S')
    end_time = datetime.strptime(data.iloc[-1]['date'] + ' ' + data.iloc[-1]['time'],
                                 '%m/%d/%Y %H:%M:%S')

    data_per_track = int((end_time - start_time).total_seconds() // interpolate_interval + 1) # interpolation interval = 2 seconds

    sampling_indices = range(0, data_per_track, SAMPLING_TIME)
    nr_sampled_data = len(sampling_indices)
    overall_data = np.full(shape=(nr_sampled_data, in_clm_len), fill_value=np.nan)
    # make output column same as input column with appending zeros to out_column_len
    target_data = np.full(shape=(nr_sampled_data, in_clm_len), fill_value=np.nan)

    temp_data = pd.DataFrame(index=range(data_per_track), columns=features, dtype=np.float)
    for slot_index in range(0, data.shape[0]):  # //
        current_time = datetime.strptime(data.iloc[slot_index]['date'] + ' ' + data.iloc[slot_index]['time'],
                                         '%m/%d/%Y %H:%M:%S')
        index1 = int((current_time - start_time).total_seconds()) // interpolate_interval
        temp_data.iloc[index1, 0:dim] = data.iloc[slot_index, 2:dim+2]

    # interpolate
    temp_data.iloc[0:sampling_indices[INPUT_LEN], :] = temp_data.iloc[0:sampling_indices[INPUT_LEN], :].interpolate(method='linear', limit_direction='forward', axis=0)
    resampled_data = temp_data.iloc[sampling_indices, 0:dim]

    # resampled_data.to_csv("Track167.csv", index=False)


    overall_data[0:nr_sampled_data, 0:dim] = temp_data.iloc[sampling_indices, 0:dim]

    temp_data_interpolated = resampled_data.interpolate(method='linear', limit_direction='forward', axis=0)
    temp_data_interpolated.to_csv("Track167_interpolated_1min.csv",index=False)
    temp_data_interpolated = np.array(temp_data_interpolated.iloc[:, 0:dim])
    aa = temp_data.iloc[sampling_indices, 0:dim]
    # aa.to_csv("Track167.csv",index=False)
    # shift from top and put on remaining columns
    for clm_nr in range(1, INPUT_LEN):
        overall_data[0: nr_sampled_data - clm_nr, clm_nr * dim:(clm_nr + 1) * dim] = overall_data[
                                                                                     1: nr_sampled_data - clm_nr + 1,
                                                                                     (clm_nr - 1) * dim:clm_nr * dim]

    return np.array(resampled_data), overall_data, temp_data_interpolated

def load_test_data_ENU(INPUT_LEN, TARGET_LEN, features,  dim, track_to_check):

    in_clm_len = INPUT_LEN*dim
    out_clm_len = TARGET_LEN * dim
    interpolate_interval = 60 # seconds
    SAMPLING_TIME = 1 # second
    data_features = ["x", "y", "cog", "sog"]
    data_dim = len(data_features)
    path = '/home/sing_sd/Desktop/anomaly_detection/PythonCode/Resources/track_pickle/'

    filename = 'track{}'.format(track_to_check)

    try:
        data = pd.read_pickle(path + filename + '.pkl')
    except IOError:
        print("Error: File does not appear to exist for track ", track_to_check)
        return 0, 0
    # without interpolation
    original_data = np.array([data.x, data.y]).transpose()

    start_time = datetime.strptime(data.iloc[0]['date'] + ' ' + data.iloc[0]['time'],
                                   '%m/%d/%Y %H:%M:%S')
    end_time = datetime.strptime(data.iloc[-1]['date'] + ' ' + data.iloc[-1]['time'],
                                 '%m/%d/%Y %H:%M:%S')

    data_per_track = int((end_time - start_time).total_seconds() // interpolate_interval + 1) # interpolation interval = 2 seconds


    sampling_indices = range(0, data_per_track, SAMPLING_TIME)
    nr_sampled_data = len(sampling_indices)
    overall_data = np.full(shape=(nr_sampled_data, in_clm_len), fill_value=np.nan)
    # make output column same as input column with appending zeros to out_column_len
    target_data = np.full(shape=(nr_sampled_data, in_clm_len), fill_value=np.nan)

    temp_data = pd.DataFrame(index=range(data_per_track), columns=data_features, dtype=np.float)
    for slot_index in range(0, data.shape[0]):  # //
        current_time = datetime.strptime(data.iloc[slot_index]['date'] + ' ' + data.iloc[slot_index]['time'],
                                         '%m/%d/%Y %H:%M:%S')
        index1 = int((current_time - start_time).total_seconds()) // interpolate_interval
        temp_data.iloc[index1, 0:data_dim] = data.iloc[slot_index, 2:data_dim+2]

    # interpolate
    temp_data.iloc[0:sampling_indices[INPUT_LEN], :] = temp_data.iloc[0:sampling_indices[INPUT_LEN], :].interpolate(method='linear', limit_direction='forward', axis=0)
    resampled_data = temp_data.iloc[sampling_indices, 0:data_dim]

    # resampled_data.to_csv("Track167.csv", index=False)

    # put lon and lat data
    overall_data[0:nr_sampled_data, 0:2] = temp_data.iloc[sampling_indices, 0:2]

    temp_data_interpolated = resampled_data.interpolate(method='linear', limit_direction='forward', axis=0)
    # temp_data_interpolated.to_csv("Track167_interpolated_1min.csv",index=False)
    temp_data_interpolated = np.array(temp_data_interpolated.iloc[:, 0:data_dim])
    # aa = temp_data.iloc[sampling_indices, 0:dim]
    # aa.to_csv("Track167.csv",index=False)
    # convert lon, lat in ENU
    lon = np.transpose(overall_data[:, 0])
    lat = np.transpose(overall_data[:, 1])
    tREF = {"lon": 12.114733,
            "lat": 54.145409,
            "ECEF": np.array([[3660725], [785776], [514624]])
            }

    zENU = WGS84toENU(lon, lat, tREF, h=0.)

    overall_data[:, 0] = np.transpose(zENU[0, :])
    overall_data[:, 1] = np.transpose(zENU[1, :])
    # put z data
    overall_data[:, 2] = np.transpose(zENU[2, :])  # insert zeros on 3rd column in data for
    # put cog and sog
    overall_data[:, 3:dim] = temp_data.iloc[sampling_indices, 2:data_dim]
    # shift from top and put on remaining columns
    for clm_nr in range(1, INPUT_LEN):
        overall_data[0: nr_sampled_data - clm_nr, clm_nr * dim:(clm_nr + 1) * dim] = overall_data[
                                                                                     1: nr_sampled_data - clm_nr + 1,
                                                                                     (clm_nr - 1) * dim:clm_nr * dim]

    return np.array(resampled_data), overall_data, temp_data_interpolated

def load_test_interpolated_data(INPUT_LEN, TARGET_LEN, features,  dim, track_to_check):
    in_clm_len = INPUT_LEN*dim
    out_clm_len = TARGET_LEN * dim
    interpolate_interval = 60 # seconds
    SAMPLING_TIME = 1
    path = '/home/sing_sd/Desktop/anomaly_detection/PythonCode/Resources/track_pickle/'

    filename = 'track{}'.format(track_to_check)

    try:
        data = pd.read_pickle(path + filename + '.pkl')
    except IOError:
        print("Error: File does not appear to exist for track ", track_to_check)
        return 0, 0
    # without interpolation
    original_data = np.array([data.x, data.y]).transpose()

    start_time = datetime.strptime(data.iloc[0]['date'] + ' ' + data.iloc[0]['time'],
                                   '%m/%d/%Y %H:%M:%S')
    end_time = datetime.strptime(data.iloc[-1]['date'] + ' ' + data.iloc[-1]['time'],
                                 '%m/%d/%Y %H:%M:%S')

    data_per_track = int((end_time - start_time).total_seconds() // interpolate_interval + 1) # interpolation interval = 2 seconds

    sampling_indices = range(0, data_per_track, SAMPLING_TIME)
    nr_sampled_data = len(sampling_indices)
    overall_data = np.full(shape=(nr_sampled_data, in_clm_len), fill_value=np.nan)
    # make output column same as input column with appending zeros to out_column_len
    target_data = np.full(shape=(nr_sampled_data, in_clm_len), fill_value=np.nan)

    temp_data = pd.DataFrame(index=range(data_per_track), columns=features, dtype=np.float)
    for slot_index in range(0, data.shape[0]):  # //
        current_time = datetime.strptime(data.iloc[slot_index]['date'] + ' ' + data.iloc[slot_index]['time'],
                                         '%m/%d/%Y %H:%M:%S')
        index1 = int((current_time - start_time).total_seconds()) // interpolate_interval
        temp_data.iloc[index1, 0:dim] = data.iloc[slot_index, 2:dim+2]

    # interpolate
    temp_data = temp_data.interpolate(method='linear', limit_direction='forward', axis=0)

    overall_data[0:nr_sampled_data, 0:dim] = temp_data.iloc[sampling_indices, 0:dim]


    # shift from top and put on remaining columns
    for clm_nr in range(1, INPUT_LEN):
        overall_data[0: nr_sampled_data - clm_nr, clm_nr * dim:(clm_nr + 1) * dim] = overall_data[
                                           1: nr_sampled_data - clm_nr + 1, (clm_nr - 1) * dim:clm_nr * dim]

    target_data[0:nr_sampled_data - INPUT_LEN - TARGET_LEN + 1, 0:out_clm_len] = overall_data[
                 INPUT_LEN: nr_sampled_data - TARGET_LEN + 1, 0:out_clm_len]

    # clear last few rows that does not have sufficient data to make ...
    overall_data[nr_sampled_data - INPUT_LEN + 1: nr_sampled_data] = np.nan

    overall_data = overall_data[~np.isnan(overall_data[:, 0])]
    target_data = target_data[~np.isnan(target_data[:, 0])]
    target_data = np.nan_to_num(target_data)
    return original_data, overall_data, target_data


# one hot encode sequence
def one_hot_encode(sequence, n_unique):
    encoding = list()
    for value in sequence:
        vector = [0 for _ in range(n_unique)]
        vector[value] = 1
        encoding.append(vector)
    return array(encoding)


# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]


# prepare data for the LSTM
def get_input_output(dim, INPUT_LEN, PRED_LEN):
    n_unique = 360
    in_clm_len = dim * INPUT_LEN
    out_clm_len = dim * PRED_LEN
    # generate random sequence
    sequence_in, sequence_out = load_all_data(dim, INPUT_LEN, PRED_LEN)

    # one hot encode
    X = one_hot_encode(sequence_in, n_unique)
    y = one_hot_encode(sequence_out, n_unique)
    # reshape as 3D
    X = X.reshape((1, X.shape[0], X.shape[1]))
    y = y.reshape((1, y.shape[0], y.shape[1]))
    return X, y