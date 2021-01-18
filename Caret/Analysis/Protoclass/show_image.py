#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 00:51:59 2020

@author: bemayer
"""


import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

pix = pd.read_csv('representative_digits.csv')

# Image output
for line in range(pix.shape[0]):
	palette_RGB = [[int(255-x*255/6)]*3 for x in range(7)]
	image_RGB = np.array([[palette_RGB[pix.iloc[line, i+15*j]] for i in range(15)]
						for j in range(16)], dtype=np.uint8)
	image = Image.fromarray(image_RGB)
	image.save('./rep_img_' + str(line) + '.jpg', 'JPEG')
