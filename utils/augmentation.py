# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Image augmentation functions
"""
import random
import cv2
import numpy as np



def random_perspective(data, degrees=10, translate=.1, scale=.1, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    rgb = data['rgb']
    height = rgb.shape[0] + border[0] * 2  # shape(h,w,c)
    width = rgb.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -rgb.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -rgb.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    # P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    # P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    # S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    # S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        for r in data.keys():
            im = data[r]
            data[r] = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=0)

    return data

def flipud(data):
    for r in data.keys():
        data[r] = np.flipud(data[r])
    return data

def fliplr(data):
    for r in data.keys():
        data[r] = np.fliplr(data[r])
    return data
