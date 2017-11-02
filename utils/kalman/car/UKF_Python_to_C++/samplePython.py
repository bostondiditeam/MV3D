import ctypes
import numpy as np

# create a reference to the C++ shared object
# IMPORTANT: Change the filepath to SharedLib.so
ukf = ctypes.cdll.LoadLibrary("./SharedLib.so")

# instantiate the UKF class
ukfInstance = ukf.ClassA()

# **************************************************************************************************************
# list of measurements
# Here are listed 6 successive example measurements. 
# In the actual code, the following list of measurements should be replaced by an actual stream of measurements.

## NOTE ON MEASUREMENT FORMAT:
# Each LiDAR measurement is an np double data type array with four elements.
# The four elements are timestamp (in us), sensor type (0 for LiDAR), px and py in that order.
# Each Radar measurement is an np double data type array with five elements.
# The five elements are timestamp (in us), sensor type (1 for Radar), rho, phi and rho dot in that order. 

meas_in_lidar1 = np.ones((4, 1), dtype = np.double)
meas_in_lidar1[0] = 1477010443000000		# timestamp
meas_in_lidar1[1] = 0						# sensor type
meas_in_lidar1[2] = 3.122427e-01			# measured px
meas_in_lidar1[3] = 5.803398e-01			# measured py

meas_in_radar1 = np.ones((5, 1), dtype = np.double)
meas_in_radar1[0] = 1477010443050000		# timestamp
meas_in_radar1[1] = 1						# sensor type
meas_in_radar1[2] = 1.014892e+00			# measured rho
meas_in_radar1[3] = 5.543292e-01			# measured phi
meas_in_radar1[4] = 4.892807e+00			# measured rho dot

meas_in_lidar2 = np.ones((4, 1), dtype = np.double)
meas_in_lidar2[0] = 1477010443100000		# timestamp
meas_in_lidar2[1] = 0						# sensor type
meas_in_lidar2[2] = 1.173848e+00			# measured px
meas_in_lidar2[3] = 4.810729e-01			# measured py

meas_in_radar2 = np.ones((5, 1), dtype = np.double)
meas_in_radar2[0] = 1477010443150000		# timestamp
meas_in_radar2[1] = 1						# sensor type
meas_in_radar2[2] = 1.047505e+00			# measured rho
meas_in_radar2[3] = 3.892401e-01			# measured phi
meas_in_radar2[4] = 4.511325e+00			# measured rho dot

meas_in_lidar3 = np.ones((4, 1), dtype = np.double)
meas_in_lidar3[0] = 1477010443200000		# timestamp
meas_in_lidar3[1] = 0						# sensor type
meas_in_lidar3[2] = 1.650626e+00			# measured px
meas_in_lidar3[3] = 6.246904e-01			# measured py

meas_in_radar3 = np.ones((5, 1), dtype = np.double)
meas_in_radar3[0] = 1477010443250000		# timestamp
meas_in_radar3[1] = 1						# sensor type
meas_in_radar3[2] = 1.698300e+00			# measured rho
meas_in_radar3[3] = 2.982801e-01			# measured phi
meas_in_radar3[4] = 5.209986e+00			# measured rho dot

# **************************************************************************************************************

## EXPLANATORY NOTES

# The UKF class and its methods are applied below on the measurements listed above.
# The order of measurements is lidar1, radar1, lidar2, radar2, lidar3, radar3.

# Six iterations of the six measurements above are shown.
# In the final implementation, the iterations hard coded below should be replaced by a loop structure to accept
# measurements as they arrive before calling the predict and the update methods.

# Each iteration comprises first of the predict method and then of the update method.

# After each call to the predict method, the predicted UKF state variables are stored in the np array predictedState.
# Note that predictedState must be passed as a void * to the predict method whose C++ implementation then populates predictedState.

# After each call to the update method, the updated UKF state variables are stored in the np array updatedState.
# Note that updatedState must be passed as a void * to the update method whose C++ implementation then populates updatedState.

# The predict method takes in as input delta_t which is the difference in micro-seconds between now and the previous timestamp.
# Note that the division by 1,000,000 needed for delta_t is done in C++, not here.

# The update method takes in as input a measurement np array in the format described above.

# The initializeStateVector method is called only when the first measurement is received. It is never used after that.

# The first input argument to every UKF method must be "ukfInstance".

# All UKF paramters such as the process noise, standard deviations etc. are defined in the C++ file classA.h.
# If any parameter is modified, the C++ code must be re-compiled with the provided Makefile.


# Initialize the predictedState vector to be all zeros. This is the variable where the predicted state after each iteration is saved.
predictedState = np.zeros((5, 1), dtype = np.double)

# Initialize the updatedState vector to be all zeros. This is the variable where the updated state after each iteration is saved.
updatedState = np.zeros((5, 1), dtype = np.double)




# Iteration 0
# Initialize the UKF with the first measurement. This method is never subsequently called.
ukf.initializeStateVector(ukfInstance, ctypes.c_void_p(meas_in_lidar1.ctypes.data)) # called only for the first measurement

# IMPORTANT: The difference in the two timestamps (delta_t) is divided by 1,000,000 in C++, not here.

# Iteration 1
timeDiff = meas_in_radar1[0] - meas_in_lidar1[0]
delta_t = timeDiff * np.ones(1, dtype = np.double)
ukf.predict(ukfInstance, ctypes.c_void_p(predictedState.ctypes.data), ctypes.c_void_p(delta_t.ctypes.data))
print "Predicted State: \n", predictedState
print "********************************"
ukf.update(ukfInstance, ctypes.c_void_p(meas_in_radar1.ctypes.data), ctypes.c_void_p(updatedState.ctypes.data))
print "Updated State: \n", updatedState
print "********************************"
 

# Iteration 2
timeDiff = meas_in_lidar2[0] - meas_in_radar1[0]
delta_t = timeDiff * np.ones(1, dtype = np.double)
ukf.predict(ukfInstance, ctypes.c_void_p(predictedState.ctypes.data), ctypes.c_void_p(delta_t.ctypes.data))
print "Predicted State: \n", predictedState
print "********************************"
ukf.update(ukfInstance, ctypes.c_void_p(meas_in_lidar2.ctypes.data), ctypes.c_void_p(updatedState.ctypes.data))
print "Updated State: \n", updatedState
print "********************************"


# Iteration 3
timeDiff = meas_in_radar2[0] - meas_in_lidar2[0]
delta_t = timeDiff * np.ones(1, dtype = np.double)
ukf.predict(ukfInstance, ctypes.c_void_p(predictedState.ctypes.data), ctypes.c_void_p(delta_t.ctypes.data))
print "Predicted State: \n", predictedState
print "********************************"
ukf.update(ukfInstance, ctypes.c_void_p(meas_in_radar2.ctypes.data), ctypes.c_void_p(updatedState.ctypes.data))
print "Updated State: \n", updatedState
print "********************************"


# Iteration 4
timeDiff = meas_in_lidar3[0] - meas_in_radar2[0]
delta_t = timeDiff * np.ones(1, dtype = np.double)
ukf.predict(ukfInstance, ctypes.c_void_p(predictedState.ctypes.data), ctypes.c_void_p(delta_t.ctypes.data))
print "Predicted State: \n", predictedState
print "********************************"
ukf.update(ukfInstance, ctypes.c_void_p(meas_in_lidar3.ctypes.data), ctypes.c_void_p(updatedState.ctypes.data))
print "Updated State: \n", updatedState
print "********************************"


# Iteration 5
timeDiff = meas_in_radar3[0] - meas_in_lidar3[0]
delta_t = timeDiff * np.ones(1, dtype = np.double)
ukf.predict(ukfInstance, ctypes.c_void_p(predictedState.ctypes.data), ctypes.c_void_p(delta_t.ctypes.data))
print "Predicted State: \n", predictedState
print "********************************"
ukf.update(ukfInstance, ctypes.c_void_p(meas_in_radar3.ctypes.data), ctypes.c_void_p(updatedState.ctypes.data))
print "Updated State: \n", updatedState
print "********************************"
