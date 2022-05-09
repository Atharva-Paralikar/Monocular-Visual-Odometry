import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import time
from json import JSONEncoder


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def get_E(F):

	global K

	E = (K.T)@F@K

	## Decomposing Essential Matrix

	W = np.array([[0,-1,0],[1,0,0],[0,0,1]])

	U_e,D_e,Vt_e = np.linalg.svd(E)

	C1 = U_e[:,2]
	R1 = U_e@W@Vt_e

	C2 = -U_e[:,2]
	R2 = U_e@W@Vt_e

	C3 = U_e[:,2]
	R3 = U_e@W.T@Vt_e

	C4 = -U_e[:,2]
	R4 = U_e@W.T@Vt_e

	C = [C1,C2,C3,C4]
	R = [R1,R2,R3,R4]

	return E,C,R

def estimate_fundamental_matrix(Points_a,Points_b):

	mean_a = Points_a.mean(axis=0)
	mean_b = Points_b.mean(axis=0)
	std_a = np.sqrt(np.mean(np.sum((Points_a-mean_a)**2, axis=1), axis=0))
	std_b = np.sqrt(np.mean(np.sum((Points_b-mean_b)**2, axis=1), axis=0))

	Ta1 = np.diagflat(np.array([np.sqrt(2)/std_a, np.sqrt(2)/std_a, 1]))
	Ta2 = np.column_stack((np.row_stack((np.eye(2), [[0, 0]])), [-mean_a[0], -mean_a[1], 1]))
	Tb1 = np.diagflat(np.array([np.sqrt(2)/std_b, np.sqrt(2)/std_b, 1]))
	Tb2 = np.column_stack((np.row_stack((np.eye(2), [[0, 0]])), [-mean_b[0], -mean_b[1], 1]))

	Ta = np.matmul(Ta1, Ta2)
	Tb = np.matmul(Tb1, Tb2)
	arr_a = np.column_stack((Points_a, [1]*Points_a.shape[0]))
	arr_b = np.column_stack((Points_b, [1]*Points_b.shape[0]))
	arr_a = np.matmul(Ta, arr_a.T)
	arr_b = np.matmul(Tb, arr_b.T)

	arr_a = arr_a.T
	arr_b = arr_b.T

	arr_a = np.tile(arr_a, 3)
	arr_b = arr_b.repeat(3, axis=1)

	A = np.multiply(arr_a, arr_b)

	U, s, V = np.linalg.svd(A)
	F_matrix = V[-1]
	F_matrix = np.reshape(F_matrix, (3, 3))
	F_matrix /= np.linalg.norm(F_matrix)

	U, S, Vh = np.linalg.svd(F_matrix)
	S[-1] = 0
	F_matrix = U @ np.diagflat(S) @ Vh
	F_matrix = Tb.T @ F_matrix @ Ta
	return F_matrix

def Ransac(matches_a,matches_b):

	num_iterator = 30000
	threshold = 0.001
	best_F_matrix = np.zeros((3, 3))
	max_inlier = 0
	num_sample_rand = 8

	xa = np.column_stack((matches_a, [1]*matches_a.shape[0]))
	xb = np.column_stack((matches_b, [1]*matches_b.shape[0]))
	xa = np.tile(xa, 3)
	xb = xb.repeat(3, axis=1)
	A = np.multiply(xa, xb)

	for i in range(num_iterator):
		index_rand = np.random.randint(matches_a.shape[0], size=num_sample_rand)
		F_matrix = estimate_fundamental_matrix(matches_a[index_rand, :], matches_b[index_rand, :])
		err = np.abs(np.matmul(A, F_matrix.reshape((-1))))
		current_inlier = np.sum(err <= threshold)
		if current_inlier > max_inlier:
			best_F_matrix = F_matrix.copy()
			max_inlier = current_inlier

	err = np.abs(np.matmul(A, best_F_matrix.reshape((-1))))
	index = np.argsort(err)

	return best_F_matrix, matches_a, matches_b

def get_features(image1,image2,frame):

	orb = cv2.ORB_create(nfeatures = 2000)

	image1_keypoint,image1_descriptors = orb.detectAndCompute(image1,None)
	image2_keypoint,image2_descriptors = orb.detectAndCompute(image2,None)

	cv2.drawKeypoints(image1,image1_keypoint,image1,color = (0,255,0))
	cv2.drawKeypoints(image2,image2_keypoint,image2,color = (0,255,0))

	bf = cv2.BFMatcher_create(cv2.NORM_HAMMING,crossCheck = True)

	matches = bf.match(image1_descriptors, image2_descriptors)
	matches = sorted(matches, key = lambda x : x.distance)
	
	points_image_1 = [image1_keypoint[m.queryIdx].pt for m in matches]
	points_image_2 = [image2_keypoint[m.trainIdx].pt for m in matches]

	matches_1 = np.asarray(points_image_1)
	matches_2 = np.asarray(points_image_2)
	# write_to_file(matches_1,matches_2,frame)

	return matches_1,matches_2,image1_descriptors,image2_descriptors

# def load_features():

# 	feature_file_paths = os.listdir("./dataset/features/")
# 	feature_file_paths.sort()
# 	total_features = {}
# 	for path in feature_file_paths:
# 		with open(feature_file_paths + path, 'r') as feature_file:
# 			features = json.load(feature_file)
# 		total_features.update(features)
	
# 	print('features loaded...')
# 	return total_features

def load_poses():
    try:
        with open("./dataset/pose.txt", 'r') as pose_file:
            poses = json.load(pose_file)
    except:
        with open("./dataset/pose.txt", 'w') as pose_file:
            pose_file.write('{}')
        poses = {}

    return poses

def read_image(frame_count):
	filename = "./dataset/processedframes/" + str(frame_count)+".png"
	image = cv2.imread(filename)
	return image

# def write_to_file(features1, features2, frame):
#     with open("./dataset/features/features.txt", 'r') as feature_file:
#         features = json.load(feature_file)
    
#     features[frame] = {
#             'features1': list(features1),
#             'features2': list(features2)
#         }
#     with open(FEATURE_FILE, 'w') as feature_file:
#         json.dump(features, feature_file)

def estimate_odometry():
	global K

	# features = load_features()
	precalc_poses = load_poses() 	

	frame_count = 1
	init_point = np.array([0, 0, 0, 1])
	H = np.eye(4)
	t = np.array([0, 0, 0]).reshape(3,1)
	R = np.eye(3)
	camera_pose = np.eye(4)
	while True:
		frame1 = read_image(frame_count)
		frame2 = read_image(frame_count + 1)
		if (frame1 is None) or (frame2 is None) or (cv2.waitKey(1) == 27):
			break
		frame_name = str(frame_count)
		pose = precalc_poses.get(frame_name)

		if pose is None:
			# print("gela")
			gray_image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
			gray_image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
			points_image_1,points_image_2,image1_descriptors,image2_descriptors = get_features(gray_image1,gray_image2,frame_name)
			F,good_matches_1,good_matches_2 = Ransac(points_image_1,points_image_2)
			essential_mat, _ = cv2.findEssentialMat(good_matches_1[:, :2], good_matches_2[:, :2], focal=K[0, 0], pp=(K[0, 2], K[1, 2]), method=cv2.RANSAC, prob=0.999, threshold=0.5)
			# E,C,R = get_E(F)
			_, new_R, new_t, mask = cv2.recoverPose(essential_mat, good_matches_1[:, :2], good_matches_2[:, :2], K)
			if np.linalg.det(new_R) < 0:
				new_R = -new_R
				new_t = -new_t
			
			precalc_poses[frame_name] = list(np.column_stack((new_R, new_t)))
			with open("./dataset/pose.txt", 'w') as pose_file:
				json.dump(precalc_poses, pose_file, cls=NumpyArrayEncoder)
		else:
			pose = np.asarray(pose)
			new_R = pose[:, :3]
			new_t = pose[:, 3].reshape(3,1)

		new_pose = np.column_stack((new_R, new_t))
		new_pose = np.vstack((new_pose, np.array([0,0,0,1])))
		camera_pose = camera_pose @ new_pose
		x_coord = camera_pose[0, -1]
		z_coord = camera_pose[2, -1]

		plt.scatter(x_coord, -z_coord, color='b')
		plt.pause(0.00001)

		frm = cv2.resize(frame1, (0,0), fx=0.5, fy=0.5)
		cv2.imshow('Frame', frm)
		print(frame_count)
		frame_count += 1

	cv2.destroyAllWindows()
	plt.show()

if __name__ == '__main__':

	K = np.array([[964.829,0,643.788],[0,964.829,484.408],[0,0,1]])

	estimate_odometry()