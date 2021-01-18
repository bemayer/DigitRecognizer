# ------------------------------------------------------------------------------
# Libraries
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Data
data = pd.read_csv('data_train.csv')
pix = data.filter(regex=('pix*'))
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Image output
for line in range(pix.shape[0]):
	palette_RGB = [[int(255-x*255/6)]*3 for x in range(7)]
	image_RGB = np.array([[palette_RGB[pix.iloc[line, i+15*j]] 
				for i in range(15)] for j in range(16)], dtype=np.uint8)
	image = Image.fromarray(image_RGB)
	image.save('./Images/img_' + str(line) + '.jpg', 'JPEG')
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Latex ouput - one case
line = 1185
print('\\begin{array}{ccccccccccccccc}')
for j in range(16):
	for i in range(15):
		print('\cellcolor[RGB]{',
			str(int(255 - 128*pix.iloc[line, i+15*j]/6)), ', ',
			str(int(255 - 128*pix.iloc[line, i+15*j]/6)), ', ',
			str(int(255 - 128*pix.iloc[line, i+15*j]/6)), '} ',
			pix.iloc[line, i+15*j],
			sep='', end='')
		if i != 14:
			print(' & ', end='')
	if (j != 15):
		print(' \\\\', end='')
	print()
print('\\end{array}')
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Latex ouput - average off all columns
pix_avg = pix.mean()
print('\\begin{array}{ccccccccccccccc}')
for j in range(16):
	for i in range(15):
		print('\cellcolor[RGB]{', str(int(255 - 128*pix_avg[i+15*j]/6)), ', ',
			str(int(255 - 128*pix_avg[i+15*j]/6)), ', ',
			str(int(255 - 128*pix_avg[i+15*j]/6)), '} ',
			round(pix_avg[i+15*j], 1),
			sep='', end='')
		if i != 14:
			print(' & ', end='')
	if (j != 15):
		print(' \\\\', end='')
	print()
print('\\end{array}')
# ------------------------------------------------------------------------------