import cv2


def main():
	path = ""
	frame_count = 1
	cap = cv2.VideoCapture("./newdataset.mp4")

	while cap.isOpened():
		ret,frame = cap.read()
		if ret:
			cv2.imwrite("./dataset/processedframes/"+str(frame_count)+".png",frame)
		frame_count += 1 
	cap.release()
	cv2.destroyAllWindows()	
if __name__ == '__main__':
	main()
