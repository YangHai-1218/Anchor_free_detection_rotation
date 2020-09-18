import cv2
import glob
import numpy as np

image_root = '/Volumes/hy_mobile/03data/DOTA-v1.5/min_split_train'
image_file_paths = glob.glob(image_root+'/*.png')

mean = 0
std = 0
means = []
stds = []
for i, image_path in enumerate(image_file_paths):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # BGR format

    means.append(np.mean(img, axis=0).mean(axis=0))
    stds.append(np.std(img, axis=0).mean(axis=0))

means = np.reshape(means, (-1, 3))
stds = np.reshape(stds, (-1, 3))

mean = np.mean(means, axis=0)
std = np.mean(stds, axis=0)

if img.dtype == np.uint8:
    mean = mean / 255
    std = std / 255
elif img.dtype == np.uint16:
    mean = mean / (2**16-1)
    std = std / (2**16-1)
else:
    print('not supported image type')
print(f'mean(BGR) : {np.round(mean,4)},  std(BGR) : {np.round(std,4)}')