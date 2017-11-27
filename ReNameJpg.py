#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:58:29 2017

@author: cooperjack
"""

import os
base_path = '/Users/cooperjack/Documents/TestImage/Output/'

import os
import os.path
img_types = ['.jpg', '.tif']
for dirpath, dirnames, fnames in os.walk(base_path):
    imgs = [f for f in fnames if os.path.splitext(f)[1] in img_types]
    for j, im in enumerate(imgs):
        name, ext = os.path.splitext(im)
        os.rename(os.path.join(dirpath, im), os.path.join(dirpath,'img_{}{}'.format(j, ext)))