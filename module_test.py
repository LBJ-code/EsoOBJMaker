import numpy as np
import cv2
import pydicom

CT_row = pydicom.read_file(r'C:\Users\Planck\Desktop\3Dsyokudo\Image005').pixel_array
CT_row = np.where(CT_row == 0, 0, CT_row - np.min(CT_row[CT_row != 0]))
tmp = np.array(255 * (CT_row / np.max(CT_row)), dtype=np.uint8)
CT_img = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
show_CT = np.copy(CT_img)
cv2.imwrite("original.png", show_CT)
cv2.imshow('original', CT_img)
cv2.waitKey(1)

# まんなかは(237, 244)
# 半径は仮に10にする．順番は↓から順番に反時計回り

iP = np.array([[237, 254], [247, 244], [237, 234], [227, 244]], dtype=np.float32)
for point in iP:
    cv2.circle(CT_img, (point[0], point[1]), 3, (200, 0, 0), -1)
    cv2.imshow('point', CT_img)
    cv2.waitKey(200)

iiP = np.array([[237, 264], [257, 244], [237, 224], [217, 244]], dtype=np.float32)
for point in iiP:
    cv2.circle(CT_img, (point[0], point[1]), 3, (0, 200, 0), -1)
    cv2.imshow('point', CT_img)
    cv2.waitKey(200)

matches = list()
for i in range(12):
    matches.append(cv2.DMatch(i, i, 0))

hazi = np.array([[256, 512], [512, 512], [512, 256], [512, 0], [256, 0], [0, 0], [0, 256], [0, 512]], dtype=np.float32)

source = np.vstack((iP, hazi)).reshape((1, -1, 2))
target = np.vstack((iiP, hazi)).reshape((1, -1, 2))

'''
for point in hazi:
    cv2.circle(CT_img, (int(point[0]), int(point[1])), 10, (0, 0, 200), -1)
    cv2.imshow('point', CT_img)
    cv2.waitKey(200)
'''

tps = cv2.createThinPlateSplineShapeTransformer()
# estimateTransFormationはバグでソースとターゲットが逆になる
tps.estimateTransformation(target, source, matches)
ret, tshape_ = tps.applyTransformation(source)

out = tps.warpImage(show_CT)
cv2.imwrite("distorted.png", out)

cv2.imshow("out", out)
cv2.waitKey(0)
