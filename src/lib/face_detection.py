#! /usr/bin/env python
#-*- coding: utf-8 -*-

"""
author  : Adrien Lafage\n
date    : february 2020
Face Detection class
===========
"""

############| IMPORTS |#############
import cv2
import torch
import numpy as np
from skimage import transform
from torchvision.transforms import functional as F
####################################

# TO CHANGE ACCORDINGLY
PATH = '/home/eisti/Private/Coding/emotion-prediction/src/lib/'

class FaceDetection:
    """ Face detection """
    def __init__(self, rescale_val=68, random_crop_val=64):
        self.faceCascade = cv2.CascadeClassifier(PATH+'haarcascade_frontalface_default.xml')
        self.rescale_val = rescale_val
        self.random_crop_val = random_crop_val

    def main(self, frame):
        """
        """
        pictures = []

        imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.faceCascade.detectMultiScale(
            imgray,
            scaleFactor=1.3,
            minNeighbors=4,
            minSize=(30,30)
        )

        # size fix
        fix = 20

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x-fix,y-fix),(x+w+fix, y+h+fix), (0,255,0),2)
            picture = imgray[y-fix:y+h+fix,x-fix:x+w+fix]

            img = np.array(picture)

            # Rescale
            h, w = img.shape[:2]
            try:
                if isinstance(self.rescale_val, int):
                    if h > w:
                        new_h, new_w = self.rescale_val * h / w, self.rescale_val
                    else:
                        new_h, new_w = self.rescale_val, self.rescale_val * w / h
                else:
                    new_h, new_w = self.rescale_val

                new_h, new_w = int(new_h), int(new_w)

                img = transform.resize(img, (new_h, new_w))

                # RandomCrop
                h, w = img.shape[:2]

                if isinstance(self.random_crop_val, int):
                    self.random_crop_val = (self.random_crop_val, self.random_crop_val)
                else:
                    assert len(self.random_crop_val) == 2
                    self.random_crop_val = self.random_crop_val
                
                new_h, new_w = self.random_crop_val

                top = np.random.randint(0, h - new_h)
                left = np.random.randint(0, w - new_w)

                img = img[top: top + new_h,
                            left: left + new_w]

                # ToTensor
                img = F.to_tensor(img)

                pictures.append(img)
            
            except:
                pass

        return(frame, pictures)