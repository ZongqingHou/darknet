from ctypes import *

class SK(Structure):
	_fields_ = [("shape", c_int),
				("points", c_int),
				("offset", c_int),
				("scale", c_float),
				("sk_dtils", POINTER(c_float))]


lib_darknet = CDLL("libdarknet.so", RTLD_GLOBAL)

load_network = lib_darknet.load_network
load_network.argtypes = [c_char_p, c_char_p, c_int]
load_network.restype = c_void_p

set_batch_network = lib_darknet.set_batch_network
set_batch_network.argtypes = [c_void_p, c_int]

free_memory = lib_darknet.free_memory
free_memory.argtypes = [SK]

openpose_forward = lib_darknet.openpose_forward
openpose_forward.argtypes = [c_void_p, POINTER(c_ubyte), POINTER(c_long), POINTER(c_long)]
openpose_forward.restype = SK


if __name__ == '__main__':
	import time
	import cv2

	net = load_network("/home/hdd/dwgit/deepwalk_camp/lib/exercise/openpose.cfg".encode("ascii"), "/home/hdd/dwgit/deepwalk_camp/lib/exercise/openpose.weight".encode("ascii"), 0)
	set_batch_network(net, 1)

	dd = cv2.imread("123.jpg")
	start_time = time.time()

	td = dd.ctypes.data_as(POINTER(c_ubyte))
	tt = openpose_forward(net, td, dd.ctypes.shape, dd.ctypes.strides)
	human_index_list = {}
	for human_index in range(tt.shape):
		for point_index in range(tt.points):
			print(tt.points)
			tmp_index = (human_index * tt.points + point_index) * tt.offset	
			if tt.sk_dtils[tmp_index + 2] > 0.05:
				cv2.circle(dd, (int(tt.sk_dtils[tmp_index] * tt.scale), int(tt.sk_dtils[tmp_index + 1] * tt.scale)), 3, (55, 255, 155), thickness=3, lineType=8, shift=0)

				print('!!!!!!!!!!!!!!!!!!!!')
				print(point_index)
				print(int(tt.sk_dtils[tmp_index] * tt.scale))
				print(tt.scale)
				print('--------------------')



	forward_time = time.time()
	print("time is %s" %(forward_time - start_time))

	# print(tt)
	free_memory(tt)
	cv2.imshow('ss', dd)
	cv2.waitKey(0)
