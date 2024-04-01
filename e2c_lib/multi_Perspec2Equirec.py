import os
import sys
import cv2
import numpy as np
import e2c_lib.Perspec2Equirec as P2E

from matplotlib import pyplot as plt

class Perspective:
    def __init__(
        self, img_array , F_P_T_array, channel=3,
        interpolation=cv2.INTER_CUBIC, average=True
    ):
        
        assert len(img_array)==len(F_P_T_array)
        
        self.img_array = img_array
        self.F_P_T_array = F_P_T_array
        self.channel = channel
        self.interpolation = interpolation
        self.average = average

    def GetEquirec(self, height:int=512, width:int=1024):
        #
        # PHI is left/right angle, THETA is up/down angle, both in degree
        #
        merge_image = np.zeros((height, width, self.channel))
        merge_mask = np.zeros((height, width, self.channel))

        for img, [F,P,T] in zip (self.img_array, self.F_P_T_array):
            per = P2E.Perspective(img, F, P, T, channel=self.channel, interpolation=self.interpolation)        # Load equirectangular image
            img , mask = per.GetEquirec(height,width)   # Specify parameters(FOV, theta, phi, height, width)
            if self.average:
                merge_image += img
            else:
                merge_image = np.where(merge_image==0, img, merge_image)
            merge_mask +=mask

        if self.average:
            merge_mask = np.where(merge_mask==0,1,merge_mask)
            merge_image = (np.divide(merge_image,merge_mask))
        else:
            merge_mask = np.where(merge_mask>0,1,0)

        return merge_image, merge_mask







