"""
Train one of my models

Usage: 
    nick_train.py --tub=<tub_path> --model=<model_path> --type=<model_type>

Options:
    -h --help    Show this screen.
"""

import time
import json
from docopt import docopt
from train import collate_records, MyCPCallback, on_best_model
from donkeycar.utils import *
from donkeycar.parts.datastore import TubHandler
import donkeycar as dk
from nick_pilots import KerasStreamline
import keras
import cv2

def nick_train(cfg, tub_path, model_path, model_type):
    # get all json files
    
    assert(model_type == 'streamline')
    kl = KerasStreamline()

    records = gather_records(cfg, tub_path)
    
    angles = []
    throttles = []
    img_arr = []

    # iterate through each record
    for record_path in records:
        #print(record_path)
        basepath = os.path.dirname(record_path)
        try:
            with open(record_path, 'r') as fp:
                json_data = json.load(fp)
        except:
            raise Exception("cannot find a record file")
    
        image_filename = json_data['cam/image_array']
        image_path = os.path.join(basepath, image_filename)
        
        angles.append(float(json_data['user/angle']))
        throttles.append(float(json_data['user/throttle']))
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = img.reshape(img.shape + (1,))
        img_arr.append(img)

    X = [np.array(img_arr).reshape(len(records), 20, 100, 1)]
    y = [np.array(angles), np.array(throttles)]
    #X.append(img_arr)
    #y.append([angle, throttle])
    
    #save_best = MyCPCallback(send_model_cb=on_best_model,
    #                         filepath=model_path,
    #                         monitor='val_loss',
    #                         verbose=True,
    #                         save_best_only=True,
    #                         mode='min',
    #                         cfg=cfg)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=cfg.MIN_DELTA,
                                               verbose=True,
                                               mode='auto')

    callbacks_list = [early_stop]
    kl.model.fit(X, y, batch_size=cfg.BATCH_SIZE, epochs=50, verbose=True)

    print('\n\n am done')
    kl.model.save(model_path)

if __name__ == "__main__":
    cfg = dk.load_config()
    args = docopt(__doc__)
    tub_path = args['--tub']
    model_path = args['--model']
    model_type = args['--type']
    nick_train(cfg, tub_path, model_path, model_type)
