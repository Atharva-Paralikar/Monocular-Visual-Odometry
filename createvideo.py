import cv2

def main():
	path = "./docs/plot/"
	frame_count = 1
	image = cv2.imread(path + "1.png")
	h,w,l = image.shape
	size = (w,h)
	video = cv2.VideoWriter("./plot.mp4",cv2.VideoWriter_fourcc(*'mp4v'),60,size)
	while (frame_count < 3720):
		image = cv2.imread(path + str(frame_count)+".png")
		video.write(image)
		print(frame_count)
		frame_count += 1
	video.release()

if __name__ == '__main__':
	main()