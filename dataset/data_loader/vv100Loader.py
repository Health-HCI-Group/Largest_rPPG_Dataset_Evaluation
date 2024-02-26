""" The dataloader for Compress datasets.

"""

import os
import cv2
import glob
import numpy as np
import json
import h5py

from scipy.interpolate import UnivariateSpline
from .BaseLoader import BaseLoader
from multiprocessing import Pool, Process, Value, Array, Manager
from tqdm import tqdm
import pandas as pd
import scipy.io as sio
import sys
import itertools
from warnings import simplefilter
# from dataset.data_loader.process_tool import *
from scipy.signal import butter, filtfilt
import csv

class vv100Loader(BaseLoader):
    """The data loader for the Compress dataset."""

    def __init__(self, name, data_path, config_data):
        self.info = config_data.INFO
        super().__init__(name, data_path, config_data)

    def get_raw_data(self, data_path):
        """Returns data directories under the path(For Compress dataset)."""

        data_dirs = glob.glob(data_path + os.sep + "*.mp4")
        if not data_dirs:
            raise ValueError(self.dataset_name + " data paths empty!")
        dirs = list()
        for data_dir in data_dirs:
            root, extension = os.path.splitext(data_dir)       # without extension
            index = os.path.split(root)[-1]                    # filename
            subject = index[0:-2]                              # filename without number
            dirs.append({"index": index, "path": data_dir, "subject": subject})
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split with begin and end values, 
        and ensures no overlapping subjects between splits"""

        # return the full directory
        if begin == 0 and end == 1:
            return data_dirs

        # get info about the dataset: subject list and num vids per subject
        data_info = dict()
        for data in data_dirs:
            subject = data['subject']
            data_dir = data['path']
            index = data['index']
            # creates a dictionary of data_dirs indexed by subject number
            if subject not in data_info:  # if subject not in the data info dictionary
                data_info[subject] = []  # make an emplty list for that subject
            # append a tuple of the filename, subject num, trial num, and chunk num
            data_info[subject].append({"index": index, "path": data_dir, "subject": subject})

        subj_list = list(data_info.keys())  # all subjects by number ID (1-27)
        subj_list = sorted(subj_list)
        num_subjs = len(subj_list)  # number of unique subjects

        # get split of data set (depending on start / end)
        choose_range = range(int(begin * num_subjs), int(end * num_subjs))
        data_dirs_new = []
        for i in choose_range:
                subj_num = subj_list[i]
                subj_files = data_info[subj_num]
                data_dirs_new += subj_files
        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """ Invoked by preprocess_dataset for multi_process. """
        filename = data_dirs[i]['subject']
        file_path = data_dirs[i]['path']
        fileindex = data_dirs[i]['index']

        # Read Frames
        # frames = self.read_video(file_path)
        frames, n, fps = self.read_video(file_path)

        # Read Labels
        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        else:
            root, extension = os.path.splitext(file_path)
            ts, bvps = self.read_wave(
                os.path.dirname(root), fileindex)

        j = int(fileindex[-1])
        information = self.read_information(os.path.dirname(root), filename, j)
        gender, fitzpatrick, location, facial_movement, talking = self.get_information(information)
        saved_filename = fileindex + f'_G{gender}_FI{fitzpatrick}_L{location}_FA{facial_movement}_T{talking}'
        
        # m = bvps.shape[0]
        # ts = np.arange(0, m*1000/60, 1000/60)
        
        # Read f_ts
        # f_ts, f = self.read_ts(os.path.dirname(root), fileindex)
        
        f_ts = np.arange(0, n*1000/fps, 1000/fps)
        # f_ts = f_ts[f_ts <= m*1000/60]j
        # f_num = f_ts.shape[0]
        # frames = frames[0:f_num]

        bvps_i = UnivariateSpline(ts, bvps, s=0)
        bvps = bvps_i(f_ts)
        bvps = bvps.astype(int)

        # target_length = frames.shape[0]
        # bvps = BaseLoader.resample_ppg(bvps, target_length)
        
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess, ext_fps = fps)
        input_name_list, label_name_list = self.save_multi_process(frames_clips, bvps_clips, saved_filename)
        file_list_dict[i] = input_name_list

    def load_preprocessed_data(self):
        """ Loads the preprocessed data listed in the file list."""

        file_list_path = self.file_list_path  # get list of files in
        file_list_df = pd.read_csv(file_list_path)
        inputs_temp = file_list_df['input_files'].tolist()
        inputs = []
        for each_input in inputs_temp:
            info = each_input.split(os.sep)[-1].split('_')
            gender = int(info[2][-1])
            fitzpatrick = int(info[3][-1])
            location = int(info[4][-1])
            facial_movement = int(info[5][-1])
            talking = int(info[6][-1])
            if (gender in self.info.GENDER) and (fitzpatrick in self.info.FITZPATRICK) and \
                (location in self.info.LOCATION) and (facial_movement in self.info.FACIAL_MOVEMENT) and \
                (talking in self.info.TALKING):
                inputs.append(each_input)
        if not inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error!')
        inputs = sorted(inputs)  # sort input file name list
        labels = [input_file.replace("input", "label") for input_file in inputs]
        self.inputs = inputs
        self.labels = labels
        self.preprocessed_data_len = len(inputs)

    @staticmethod
    def read_video(video_file):
        """Reads a video file, returns frames(T, H, W, 3) """
        VidObj = cv2.VideoCapture(video_file)
        VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
        success, frame = VidObj.read()
        frames = list()
        
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            # frame = np.asarray(frame)
            frames.append(frame)
            success, frame = VidObj.read()
        framess = np.asarray(frames)
        n = framess.shape[0]
        fps = VidObj.get(cv2.CAP_PROP_FPS)
        return framess, n, fps

    @staticmethod
    def read_wave(dir, filename):
        with h5py.File(os.path.join(dir,"vv100_ts_bvps.h5"), 'r') as hf:
            dataset_name = filename
            array = hf[dataset_name][:]
            return array[0], array[1]

    # @staticmethod
    # def read_ts(dir, filename):
    #     with h5py.File(os.path.join(dir,"COMPRESS_ts_frames.h5"), 'r') as hf:
    #         dataset_name = filename
    #         array = hf[dataset_name][:]
    #         return array[0], array[1]

    @staticmethod
    def read_information(dir, filename, j):
        file = filename + '.json'
        with open(os.path.join(dir,file), 'r') as f:
            data = json.load(f)
            gender = data['participant']['gender']
            fitzpatrick = data['participant']['fitzpatrick']
            location = data['location']['location']
            facial_movement = data['scenarios'][j-1]['scenario_settings']['facial_movement']
            talking = data['scenarios'][j-1]['scenario_settings']['talking']
            information = [gender, fitzpatrick, location, facial_movement, talking]
        return information

    @staticmethod
    def get_information(information):
        gender = ''
        if information[0] == 'M':
            gender = 1
        elif information[0] == 'F':
            gender = 2
        else:
            raise ValueError("Error with vv100 dataset labels! "
            "The following gender label is not supported: {0}".format(information[0]))
        
        fitzpatrick = int(information[1])
        if fitzpatrick !=1 and fitzpatrick !=2 and fitzpatrick !=3 and fitzpatrick !=4 and fitzpatrick !=5 and fitzpatrick !=6:
            raise ValueError("Error with vv100 dataset labels! "
            "The following fitzpatricking label is not supported: {0}".format(information[1]))

        location = ''
        if information[2] == 'KLL':
            location = 1
        elif information[2] == 'KU':
            location = 2
        elif information[2] == 'B': 
            location = 3
        elif information[2] == 'T': 
            location = 4
        elif  information[2] == 'KUL': 
            location = 5
        elif information[2] == 'WZC':
            location = 6
        elif information[2] == 'A':
            location = 7
        elif information[2] == 'TB': 
            location = 8
        elif information[2] == 'RT': 
            location = 9
        elif information[2] == 'KL': 
            location = 0
        else:
            raise ValueError("Error with vv100 dataset labels! "
            "The following location label is not supported: {0}".format(information[2]))

        facial_movement = ''
        if information[3] == 'No movement':
            facial_movement = 1
        elif information[3] == 'Moderate movement':
            facial_movement = 2
        elif information[3] == 'Slight movement':
            facial_movement = 3
        else:
            raise ValueError("Error with vv100 dataset labels! "
            "The following facial_movement label is not supported: {0}".format(information[5]))

        talking = ''
        if information[4] == 'N':
            talking = 1
        elif information[4] == 'Y':
            talking = 2
        else:
            raise ValueError("Error with vv100 dataset labels! "
            "The following talking label is not supported: {0}".format(information[6]))

        return gender, fitzpatrick, location, facial_movement, talking
