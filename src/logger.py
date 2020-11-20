# -*- coding: utf-8 -*-
"""
.. codeauthor:: Dominik Höchemer <dominik.hoechemer@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import csv
import os
import xml.etree.cElementTree as ET

from .io_utils import create_directory_if_not_exists


class CSVLogger(object):
    def __init__(self, path):
        super(CSVLogger, self).__init__()
        self._path = path

    def write_logs(self, logs):
        # write header
        if not os.path.exists(self._path):
            with open(self._path, 'w') as f:
                w = csv.DictWriter(f, list(logs.keys()))
                w.writeheader()

        with open(self._path, 'a') as f:
            w = csv.DictWriter(f, list(logs.keys()))
            w.writerow(logs)
