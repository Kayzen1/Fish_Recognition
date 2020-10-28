import cv2
import numpy as np
import os
# 定义保存图片函数
# image:要保存的图片名字
# addr；图片地址与相片名字的前部分
# num: 相片，名字的后缀。int 类型

file_dir = os.path.dirname(os.path.realpath(__file__))
def save_image(image,num,video_num):
	img_path = os.path.join(file_dir ,'video'+str(video_num))
	img_name = os.path.join(img_path ,'video'+str(video_num)+'_'+str(num).zfill(5) + '.jpg')
	cv2.imwrite(img_name,image)

def read_video(video_num):
	videoCapture = cv2.VideoCapture(str(video_num)+".mp4")
	# 通过摄像头的方式
	# videoCapture=cv2.VideoCapture(1)
	print(videoCapture.get(7))
	#读帧
	success, frame = videoCapture.read()
	print(frame.shape)
	i = 0
	timeF = 20
	j=0
	while success:
		i = i + 1
		if (i % timeF == 0):
			j = j + 1
			save_image(frame,j,video_num)
			if success:
				print('save image:',j)
		success, frame = videoCapture.read()

read_video(11)
read_video(1)

# for i in range(2,11):
# 	read_video(i)
