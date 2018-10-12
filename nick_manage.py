"""
Scripts to drive a donkey 2 car

Usage:
    manage.py (drive) [--model=<model>] [--js] [--type=(linear|categorical|rnn|i
mu|behavior|3d)] [--camera=(single|stereo)]
    manage.py (train) [--tub=<tub1,tub2,..tubn>] (--model=<model>) [--transfer=<
model>] [--type=(linear|categorical|rnn|imu|behavior|3d)] [--continuous] [--aug]


Options:
    -h --help     Show this screen.
    --js          Use physical joystick.
"""

import os
import time
from my_joystick import MyJoystickController
from docopt import docopt
import donkeycar as dk
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.transform import Lambda
from donkeycar.parts.throttle_filter import ThrottleFilter
from nick_pilots import KerasStreamline
from nick_utils import NickCamera
import cv2

def drive(cfg, model_path=None, model_type=None):
    V = dk.vehicle.Vehicle()
    
    assert(cfg.CAMERA_TYPE == 'PICAM')
    from donkeycar.parts.camera import PiCamera
    if (model_type == 'streamline'):
        cam = NickCamera(100, 20, True)
    else:
       cam = NickCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, grayscale=False)
     
    V.add(cam,
        outputs=['cam/image_array'],
        threaded=True)

    cont_class = MyJoystickController
    ctr = cont_class(throttle_scale = cfg.JOYSTICK_MAX_THROTTLE, 
            steering_scale=cfg.JOYSTICK_STEERING_SCALE,
            auto_record_on_throttle=cfg.AUTO_RECORD_ON_THROTTLE)

    V.add(ctr, 
            inputs=['cam/image_array'],
            outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
            threaded=True)

    def pilot_condition(mode):
        return not (mode == 'user')
    pilot_condition_part = Lambda(pilot_condition)
    V.add(pilot_condition_part, 
            inputs=['user/mode'],
            outputs=['run_pilot'])

    if model_path:
        if (model_type == 'streamline'):
            kl = KerasStreamline()
        else:
            kl = dk.utils.get_model_by_type(model_type, cfg)
        assert('.h5' in model_path)
       	start = time.time()
        print('loading model', model_path)
        kl.load(model_path)
        print('finished loading in %s sec.' % (str(time.time()-start)))


        V.add(kl,
                inputs=['cam/image_array'],
                outputs=['pilot/angle', 'pilot/throttle'],
                run_condition='run_pilot')

    def drive_mode(mode, user_angle, user_throttle, 
            pilot_angle, pilot_throttle):
        if mode == 'user':
            return user_angle, user_throttle
        elif mode == 'local_angle':
            return pilot_angle, user_throttle
        else:
            return pilot_angle, pilot_throttle

    drive_mode_part = Lambda(drive_mode)
    V.add(drive_mode_part,
            inputs=['user/mode', 'user/angle', 'user/throttle', 
                'pilot/angle', 'pilot/throttle'],
            outputs=['angle', 'throttle'])

    assert(cfg.DRIVE_TRAIN_TYPE == 'SERVO_ESC')
    from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle

    steering_controller = PCA9685(cfg.STEERING_CHANNEL, cfg.PCA9685_I2C_ADDR,
            busnum=cfg.PCA9685_I2C_BUSNUM)
    steering = PWMSteering(controller=steering_controller,
            left_pulse=cfg.STEERING_LEFT_PWM,
            right_pulse=cfg.STEERING_RIGHT_PWM)

    throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL, cfg.PCA9685_I2C_ADDR,
            busnum=cfg.PCA9685_I2C_BUSNUM)
    throttle = PWMThrottle(controller = throttle_controller,
            max_pulse=cfg.THROTTLE_FORWARD_PWM,
            zero_pulse=cfg.THROTTLE_STOPPED_PWM,
            min_pulse=cfg.THROTTLE_REVERSE_PWM)

    V.add(steering, 
            inputs=['angle'])
    V.add(throttle,
            inputs=['throttle'])

    inputs=['cam/image_array', 'user/angle', 'user/throttle', 'user/mode']
    types=['image_array', 'float', 'float', 'str']

    th = TubHandler(path=cfg.DATA_PATH)
    tub = th.new_tub_writer(inputs=inputs, types=types)
    V.add(tub,
            inputs=inputs,
            outputs=['tub/num_records'],
            run_condition='recording')
    
    print('you can now move your joystick to drive your car')
    ctr.set_tub(tub)

    V.start(rate_hz=cfg.DRIVE_LOOP_HZ,
            max_loop_count=cfg.MAX_LOOPS)

'''
def train(cfg, tub, model, transfer, model_type):
    verbose = cfg.VERBOSE_TRAIN

    gen_records = {}
    opts = {}

    kl = dk.utils.get_model_by_type(model_type, cfg)

    if transfer:
        kl.load(transfer_model)
    
    kl.compile()

    if cfg.PRINT_MODEL_SUMMARY:
        print(kl.model.summary())

    opts['keras_pilot'] = kl

    records = dk.utils.gather_records(cfg, tub_names_opts)
    print('collating %d records ...' % (len(records)))
    for record_path in records:
        basepath = os.path.dirname(record_path)
        index = dk.utils.get_record_index(record_path)
'''




if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()

    if args['drive']:
        model_type = args['--type']
        model_path = args['--model']
        drive(cfg, model_path=model_path, model_type=model_type)
        
    if args['train']:
        from train import multi_train
        tub = args['--tub']
        model = args['--model']
        transfer = args['--transfer']
        model_type = args['--type']
        continuous = args['--continuous']
        aug = args['--aug']
        multi_train(cfg, tub, model, transfer, model_type, continuous, aug)
