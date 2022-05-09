import cv2
import numpy as np
import os
from scipy.ndimage import map_coordinates as interp2

def undistort_image(image,lookuptable):

	reshaped_lut = lookuptable[:, 1::-1].T.reshape((2, image.shape[0], image.shape[1]))
	undistorted_image =  np.rollaxis(np.array([interp2(image[:, :, channel], reshaped_lut, order=1)for channel in range(0, image.shape[2])]), 0, 3)
	return undistorted_image

def read_camera_data():
	intrinsic_path = "./dataset/Oxford_dataset/model/stereo_narrow_left.txt"
	lookuptable_path ="./dataset/Oxford_dataset/model/stereo_narrow_left_distortion_lut.bin"
	intrinsics = np.loadtxt(intrinsic_path)
	fx = intrinsics[0,0]
	fy = intrinsics[0,1]
	cx = intrinsics[0,2]
	cy = intrinsics[0,3]
	G_camera_image = intrinsics[1:5,0:4]
	lookuptable = np.fromfile(lookuptable_path, np.double)
	lookuptable = lookuptable.reshape([2, lookuptable.size//2])
	LUT = lookuptable.transpose()
	return fx, fy, cx, cy, G_camera_image, LUT

def preprocess():
	fx, fy, cx, cy, G_camera_image, LUT = read_camera_data()
	camera_intrinsic_matrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
	print(camera_intrinsic_matrix)
# 	path = "./dataset/Oxford_dataset/stereo/centre"
# 	image_path = os.listdir(path)
# 	image_path.sort()
# 	frame_count = 1
# 	for path in image_path:
# 		image = cv2.imread("./dataset/Oxford_dataset/stereo/centre/"+path)
# 		undistorted_image = undistort_image(image,LUT)
# 		gray_image = cv2.cvtColor(undistorted_image,cv2.COLOR_BGR2GRAY)
# 		cropped_image = gray_image[:800,:]
# 		filename = './dataset/processedframes/' + str(frame_count)+'.png'
# 		cv2.imwrite(filename,cropped_image)
# 		frame_count += 1
# 	print("Done")

if __name__ == '__main__':
	preprocess()