<p align="center">
:heart: Please remember to :star: this repo if you find it useful and cite our work if you end up using it in your work! :heart:
</p>
<p align="center">
:heart: If you have any questions or concerns, please create an <a href="https://github.com/ubicomplab/rPPG-Toolbox/issues">issue</a> :memo:! :heart:
</p>




# :wrench: Setup

Create a conda environment with the configuration file:

`conda env create -f environment.yml`

# :computer: Examples of Neural Network Training

Please use config which has *train_and_test* in *TOOLBOX_MODE*.

Training on VitalVideo and Testing on MMPD With TSCAN.

STEP 1: Download the VitalVideo and MMPD raw data.

STEP 2: Create a HDF5 file of ppg data list from VitalVideo with running `python set_list.py`

STEP 3: Modify `./configs/vv100_train_configs/vv100_vv100_MMPD_TSCAN_BASIC.yaml`

STEP 4: Run `python main.py --config_file ./configs/vv100_train_configs/vv100_vv100_MMPD_TSCAN_BASIC.yaml`

Note 1: Set *DO PREPROCESS* to *True* on the yaml file if it is the first time. And turn it off when you train the network after the first time.

Note 2: The example yaml setting will allow 100% of VitalVideo to train and 100% of MMPD to test. 
After training, it will use the last model to test on MMPD.

# :computer: Example of Using Pre-trained Models

Please use config which has *only_test* in *TOOLBOX_MODE*.

For example, if you want to run The model trained on VitalVideo and tested on PURE.

STEP 1: Modify `./configs/vv100_test_configs/vv100_PURE_TSCAN_BASIC.yaml`

STEP 2: Run `python main.py --config_file ./configs/vv100_test_configs/vv100_PURE_TSCAN_BASIC.yaml`
