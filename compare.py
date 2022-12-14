import cv2
import matplotlib.pyplot as plt
import copy
from src import util
import numpy as np
import os

from src.body import Body
body_estimation_pt = Body()
from src.body_meg import Body
body_estimation_mge = Body()

image_folder = './data'
for i, image_name in enumerate(os.listdir(image_folder)):
    image_path = os.path.join(image_folder, image_name)
    oriImg = cv2.imread(image_path)  # B,G,R order
    # torch model inference
    candidate_pt, subset_pt = body_estimation_pt(oriImg)
    # megengine model inference
    candidate_mge, subset_mge = body_estimation_mge(oriImg)
    # differerce
    np.testing.assert_allclose(candidate_pt, candidate_mge, rtol=1e-3)
    np.testing.assert_allclose(subset_pt, subset_mge, rtol=1e-3)
    print('The {}th image named {} passed'.format(i, image_name))

print('Counting inference time......')

# time compare
iterations = 50
import time

test_image = 'images/demo.jpg'
oriImg = cv2.imread(test_image)

# torch model inference time
time_start = time.time()  # 记录开始时间
for iter in range(iterations):
    _, _ = body_estimation_pt(oriImg)

time_end = time.time()  # 记录结束时间
mean_time_pt = (time_end - time_start)/iterations
print(" Torch model inference time: {:.6f}s, FPS: {} ".format(mean_time_pt, 1/mean_time_pt))



# mge model inference time
time_start = time.time()  # 记录开始时间
for iter in range(iterations):
    _, _ = body_estimation_mge(oriImg)

time_end = time.time()  # 记录结束时间
mean_time_mge = (time_end - time_start)/iterations
print(" Megengine model inference time: {:.6f}s, FPS: {} ".format(mean_time_mge, 1/mean_time_mge))



# Uncomment the block below to view the visualization results
'''
test_image = 'images/demo.jpg'
oriImg = cv2.imread(test_image)  # B,G,R order

# torch model inference
candidate_pt, subset_pt = body_estimation_pt(oriImg)
canvas_pt = copy.deepcopy(oriImg)
canvas_pt = util.draw_bodypose(canvas_pt, candidate_pt, subset_pt)
plt.imshow(canvas_pt[:, :, [2, 1, 0]])
plt.axis('off')
plt.show()

candidate_mge, subset_mge = body_estimation_mge(oriImg)
canvas_mge = copy.deepcopy(oriImg)
canvas_mge = util.draw_bodypose(canvas_mge, candidate_mge, subset_mge)
plt.imshow(canvas_mge[:, :, [2, 1, 0]])
plt.axis('off')
plt.show()
'''