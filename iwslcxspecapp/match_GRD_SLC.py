#!/usr/bin/python
import os
import logging
import get_path_from_base_SAFE
import glob
from explodesafename import ExplodeSAFE
import datetime


def match_GRD_SLC(safenamegrd):
    if 'GRDH' in safenamegrd:
        res = safenamegrd.replace('GRDH', 'SLC_')
    return res


def match_SLC_GRD(safenameslc, type_seek='GRDH'):
    """
    return the safe if it exists at ifremer otherwise None
    """
    goodsafe = None
    mini_ecart = datetime.timedelta(seconds=500000000)
    res = safenameslc.replace('SLC_', type_seek)
    obj = ExplodeSAFE(res)
    st = obj.get('startdate')
    res_base = res[0:10] + '*.SAFE'
    logging.debug('res: %s', res)
    fp = get_path_from_base_SAFE.get_path_from_base_SAFE(res)
    base = os.path.basename(fp)
    fp = fp.replace(base, res_base)
    pot = glob.glob(fp)
    for safe in pot:
        logging.debug('potential safe %s', safe)
        basesafel1 = os.path.basename(safe)
        instance = ExplodeSAFE(basesafel1)
        l1st = instance.get('startdate')
        l1ee = instance.get('enddate')
        #         if l1st<=st and l1ee>=ee:
        if abs(st - l1st) < mini_ecart:
            mini_ecart = abs(st - l1st)
            if mini_ecart < datetime.timedelta(seconds=5):
                goodsafe = safe
    return goodsafe
