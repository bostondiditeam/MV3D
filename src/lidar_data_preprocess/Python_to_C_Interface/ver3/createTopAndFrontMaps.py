import ctypes
import numpy as np

def createTopAndFrontMaps(raw, num, top_flip, front_flip, top_paras, front_paras, c_lib_path):
	[TOP_X_MIN, TOP_X_MAX, TOP_Y_MIN, 
	TOP_Y_MAX, TOP_Z_MIN, TOP_Z_MAX, 
	TOP_X_DIVISION, TOP_Y_DIVISION, TOP_Z_DIVISION, 
	Xn, Yn, Zn] = top_paras

	[FRONT_PHI_MIN, FRONT_PHI_MAX, 
	FRONT_THETA_MIN, FRONT_THETA_MAX, 
	FRONT_PHI_DIVISION, FRONT_THETA_DIVISION, 
	Rn, Cn, Fn] = front_paras

	# create a handle to LidarPreprocess.c
	SharedLib = ctypes.cdll.LoadLibrary(c_lib_path)

	# call the C function to create top view maps
	# The np array indata will be edited by createTopViewMaps to populate it with the 8 top view maps 
	SharedLib.createTopAndFrontMaps(ctypes.c_void_p(raw.ctypes.data),
								ctypes.c_int(num),
								ctypes.c_void_p(top_flip.ctypes.data), 
								ctypes.c_void_p(front_flip.ctypes.data),
								ctypes.c_float(TOP_X_MIN), ctypes.c_float(TOP_X_MAX), 
								ctypes.c_float(TOP_Y_MIN), ctypes.c_float(TOP_Y_MAX), 
								ctypes.c_float(TOP_Z_MIN), ctypes.c_float(TOP_Z_MAX), 
								ctypes.c_float(TOP_X_DIVISION), ctypes.c_float(TOP_Y_DIVISION), ctypes.c_float(TOP_Z_DIVISION), 
								ctypes.c_int(Xn), ctypes.c_int(Yn), ctypes.c_int(Zn),
								ctypes.c_float(FRONT_PHI_MIN), ctypes.c_float(FRONT_PHI_MAX), 
								ctypes.c_float(FRONT_THETA_MIN), ctypes.c_float(FRONT_THETA_MAX), 
								ctypes.c_float(FRONT_PHI_DIVISION), ctypes.c_float(FRONT_THETA_DIVISION),
								ctypes.c_float(Cn), ctypes.c_float(Rn), ctypes.c_float(Fn)
								)	




