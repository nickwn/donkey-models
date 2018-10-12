from donkeycar.parts.camera import BaseCamera
import donkeycar as dk
import cv2
import numpy as np
class NickCamera(BaseCamera):
    def __init__(self, image_w=160, image_h=120, grayscale=False):
        self.cap = cv2.VideoCapture()
        self.image_w = image_w
        self.image_h = image_h
        if grayscale:
            self.frame = np.zeros((image_w, image_h), np.uint8)
        else:
            self.frame = np.zeros((image_w, image_h, 3), np.uint8)

    def process_image(self):
        ret, frame = self.cap.read()
        if frame == None:
            return
        frame = cv2.resize(frame, (self.image_w, self.image_h))
        if(self.grayscale):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        self.frame = frame 

    def run(self):
        # not run unless not marked as threaded
        # here for redundancy
        self.process_image()
        return self.frame

    def update(self):
        self.process_image()

    def shutdown(self):
        self.cap.release()
