"""
author: Antoine Grouazel
"""
import sys
import os
from SAFEsortingfunctions import WhichArchiveDir


def get_safe_basename_from_fullpath_measu(fullpathmeasu):
    basename_safe = os.path.basename(os.path.abspath(os.path.join(fullpathmeasu, os.path.pardir, os.path.pardir)))
    return basename_safe


def get_path_from_base_SAFE(inputa):
    if '.SAFE' not in inputa:
        inputa += '.SAFE'
    dira = WhichArchiveDir(inputa)
    final_path = os.path.join(dira, inputa)
    return final_path


if __name__ == '__main__':
    # input is supposed to be a SAFE "S1A....SAFE"
    inputa = sys.argv[1]
    final_path = get_path_from_base_SAFE(inputa)
    print(final_path)
