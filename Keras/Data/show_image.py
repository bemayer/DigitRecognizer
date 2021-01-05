#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 00:51:59 2020

@author: bemayer
"""

import numpy as np
import pandas as pd
from PIL import Image

line = 2

data=pd.read_csv("data_test_completed.csv")
pix=data.filter(regex=("pix*"))
palette_RGB = [[int(255-x*255/6)]*3 for x in range(7)]
image_RGB = np.array([[palette_RGB[pix.iloc[line,i+15*j]] for i in range(15)] 
                      for j in range(16)], dtype=np.uint8)
image = Image.fromarray(image_RGB)
image.show()
