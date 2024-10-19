import random
import cv2
import numpy as np



fig = np.zeros((729, 438, 3), np.uint8)
fig.fill(255)
cv2.ellipse(fig, (388, 181), (8, 8), 0, 140, 170, (21, 80, 240), 2, lineType=cv2.LINE_AA)
cv2.imshow('Angle Annotation', fig)
cv2.waitKey(0)
cv2.destroyAllWindows()





