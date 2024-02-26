import os
import pandas as pd
import json
import numpy as np
import glob
import h5py

# path to raw data
path_to_data = '/data1/VitalVideo/vv100'

for dirpath, dirnames, filenames in os.walk(path_to_data):

    for filename in filenames:

        if filename.endswith('.json'):

            with open(os.path.join(dirpath, filename)) as json_file:

                data = json.load(json_file)
                for scenario in data['scenarios']:

                    # Check if the recording file exists in the directory (some videos were unreadable so they had to be deleted)
                    recording_link = scenario['recordings']['RGB']['filename']
                    if scenario['recordings']['RGB']['filename'] in filenames:

                        ts_list = []
                        ppg_values_list = []
                        for item in scenario['recordings']['ppg']['timeseries']:
                            if item[0] not in ts_list:
                                ts_list.append(item[0])
                                ppg_values_list.append(item[1])
                        
                        ts_values_array = np.asarray(ts_list)
                        ppg_values_array = np.asarray(ppg_values_list)
                        ppg_array = np.vstack((ts_values_array, ppg_values_array))
                        save_filename = str(recording_link)
                        np.save(path_to_data+"/ts_bvps/"+save_filename[:-4]+".npy", ppg_array)

# path to npy
npy_folder = '/data1/VitalVideo/vv100/ts_bvps/*.npy'

# path to h5
output_hdf5_file = '/data1/VitalVideo/vv100/vv100_ts_bvps.h5'

# set a list of all of npy files
npy_files = glob.glob(npy_folder)

# set up h5 file
with h5py.File(output_hdf5_file, 'w') as hf:
    for npy_file in npy_files:
        
        # load numpy data from npy files
        np_array = np.load(npy_file)

        # Use the file name (without extension) as the dataset name
        dataset_name = npy_file.split('/')[-1].split('.')[0]
        hf.create_dataset(dataset_name, data=np_array)