
from donkeycar.parts.controller import Joystick, JoystickController


class MyJoystick(Joystick):
    #An interface to a physical joystick available at /dev/input/js0
    def __init__(self, *args, **kwargs):
        super(MyJoystick, self).__init__(*args, **kwargs)

            
        self.button_names = {
            0x220 : 'dup',
            0x221 : 'ddown',
            0x222 : 'dleft',
            0x223 : 'dright',
            0x130 : 'x',
            0x131 : 'circle',
            0x133 : 'triangle',
            0x134 : 'square',
            0x136 : 'lb',
            0x137 : 'rb',
            0x138 : 'lt',
            0x139 : 'rt',
            0x13a : 'select',
            0x13b : 'start',
            0x13d : 'ljb',
            0x13e : 'rjb',
        }


        self.axis_names = {
            0x0 : 'lj',
            0x4 : 'rj',
        }



class MyJoystickController(JoystickController):
    #A Controller object that maps inputs to actions
    def __init__(self, *args, **kwargs):
        super(MyJoystickController, self).__init__(*args, **kwargs)


    def init_js(self):
        #attempt to init joystick
        try:
            self.js = MyJoystick(self.dev_fn)
            self.js.init()
        except FileNotFoundError:
            print(self.dev_fn, "not found.")
            self.js = None
        return self.js is not None


    def init_trigger_maps(self):
        #init set of mapping from buttons to function calls
            
        self.button_down_trigger_map = {
            'triangle' : self.erase_last_N_records,
            'x' : self.emergency_stop,
            'circle' : self.toggle_manual_recording,
            'select' : self.toggle_mode,
            'start' : self.toggle_constant_throttle,
            'dup' : self.increase_max_throttle,
            'ddown' : self.decrease_max_throttle,
        }


        self.axis_trigger_map = {
            'lj' : self.set_steering,
            'rj' : self.set_throttle,
        }


