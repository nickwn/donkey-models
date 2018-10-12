"""
Cvt tub to grayscale

Usage:
    rgb2gray.py --from=<tub> --to=<tub>

Options:
    -h --help   Show this screen.
"""

import os
import time
from docopt import docopt
from train import gather_records, collate_records
from donkeycar.utils import *
from donkeycar.parts.datastore import TubHandler
import donkeycar as dk
import cv2

def rgb2gray_tub(cfg, fr, to):
    opts = {}
    gen_records = {}
    opts['categorical'] = True #type(kl) in [KerasCategorical, KerasBehavioral]
    records = gather_records(cfg, fr, opts, verbose=True)
    print('collating %d records ...' % (len(records)))
    collate_records(records, gen_records, opts)

    tub_data = {}
    inputs_img = []
    gray_img = []
    angles = []
    throttles = []

    inputs=['cam/image_array',
            'user/angle',
            'user/throttle',
            'user/mode']
    types=['image_array',
           'float', 'float',
           'str']
    th = TubHandler(path=cfg.DATA_PATH)
    tub = th.new_tub_writer(inputs=inputs, types=types)
    
    for key, record in gen_records.items():
        #print(record)
        if record['img_data'] is None:
            filename = record['image_path']
            img_arr = load_scaled_image_arr(filename, cfg)

            if img_arr is None:
                break

            record['img_data'] = img_arr

        else:
            img_arr = record['img_data']

        tub_data = {
            'cam/image_array': cv2.resize(rgb2gray(img_arr), (100, 20)),
            'user/angle': record['angle'],
            'user/throttle': record['throttle'],
            'user/mode': record['json_data']['user/mode']
        }
        
        #print(tub_data)
        tub.put_record(tub_data)

if __name__ == "__main__":
    cfg = dk.load_config()
    args = docopt(__doc__)
    fr = args['--from']
    to = args['--to']
    rgb2gray_tub(cfg, fr, to)
