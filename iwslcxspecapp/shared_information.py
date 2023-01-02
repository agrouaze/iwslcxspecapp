# coding=utf-8
"""
purpose: file to share information about S-1 mission used in many scripts
Author: Antoine Grouazel
"""
import datetime
import os
import logging

PROJECT_DIR_DATARMOR = '/home/datawork-cersat-public/cache/project/mpc-sentinel1/'
PROJECT_DIR_DATARMOR_ALT = '/home/datawork-cersat-public/project/mpc-sentinel1/'
datarmor_archive_esa_ifremer = os.path.join(PROJECT_DIR_DATARMOR_ALT, "data", "esa")
# UNTARED_SPOOL = os.path.join(PROJECT_DIR,'workspace','spool_untared')
data_dir_s1a = os.path.join(PROJECT_DIR_DATARMOR, 'data', 'esa', 'sentinel-1a')
data_dir_s1b = os.path.join(PROJECT_DIR_DATARMOR, 'data', 'esa', 'sentinel-1b')
dir_aux_as_it_is_download = os.path.join(PROJECT_DIR_DATARMOR_ALT, 'data', 'auxiliary')
dir_data = {'S1A': data_dir_s1a,
            'S1B': data_dir_s1b}
ancillary_dataset = os.path.join(PROJECT_DIR_DATARMOR, "data", "ancillary")
mission_start_s1a = datetime.datetime(2014, 4, 6)
mission_start_s1b = datetime.datetime(2016, 4, 25)  # "first iw received
# Avant que S1B rejoigne son orbite definitive, les cycles ne faisaient pas 12 jours.
# Le premier cycle complet est le 10 et a commence le 14/06/2017 vers 18.00 apres l'orbite 152.
start_missions = {
    'S1A': mission_start_s1a,
    'S1B': mission_start_s1b}
first_s1b_wv = datetime.datetime(2016, 6, 15)  # WV valid Level2
first_s1a_wv = datetime.datetime(2015, 1,
                                 30)  # WV valid Level2 changed by agrouaze the 4dec 2018 (before was set to first acquisition Level1 WV)
first_valid_data = {'S1A': first_s1a_wv,
                    'S1B': first_s1b_wv}
s1a_calibration_date_WV = datetime.datetime(2014, 12, 12)
s1a_good_data_wv_calibrated = datetime.datetime(2016, 4, 18)


TYPES = ["OCN_", "GRDH", "SLC_", "GRDM", "GRDF", "RAW_"]
WORKING_DIR = os.path.join(PROJECT_DIR_DATARMOR_ALT, 'workspace')
sats_acro = {'S1A': 'sentinel-1a'
    , 'S1B': 'sentinel-1b'}
sats_full = {'sentinel-1a': 'S1A',
             'sentinel-1b': 'S1B'}
MODES = ['s1', 's2', 's3', 's4', 's5', 's6', 'wv', 'ew', 'iw']
macro_MODES = ['SM', 'WV', 'IW', 'EW']



def give_me_level_from_type(type_format):
    """
    type_format (str): ex: GRDH
    """
    if 'RAW' in type_format:
        level = 'L0'
    elif 'OCN' in type_format:
        level = 'L2'
    else:
        level = 'L1'
    return level


def give_me_prod_uploaded_at_pebsco(write=False):
    list_prod = []
    for sat in ['S1A', 'S1B']:
        #     for tprod in ['GRDH_1S','GRDF_1S','GRDM_1S','SLC__1S']:
        for tprod in ['GRDH_1S']:
            for mode in ['SM', 'IW', 'EW']:
                #         for mode in ['SM','IW','EW','WV']:
                list_prod.append(sat + '_' + mode + '_' + tprod)
    list_prod.append('S1A_WV_SLC__1S')
    list_prod.append('S1B_WV_SLC__1S')
    if write:
        filout = os.path.join(PROJECT_DIR_DATARMOR, "workspace", "syntool_upload_pebso/config_upload.txt")
        fid = open(filout, 'w')
        for dd in list_prod:
            fid.write(dd + '\n')
        fid.close()
        logging.info(filout)
    return list_prod


QUARANTINE = {
    # 'mpc':os.path.join(PROJECT_DIR,"workspace","quarantine/"),
    'datarmor_mpc': os.path.join(PROJECT_DIR_DATARMOR, "workspace", "quarantine/")
}