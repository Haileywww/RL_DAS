#!/usr/bin/python
import numpy as np


# Assumes points are between 0 and 1 
def tile_points(points, N):
	x_min, x_max = 48, 62
	y_min, y_max = -7, 5
	x_length = x_max - x_min
	y_length = y_max - y_min

	# 9 tiles surrounding individual coordinates
	point_tile = np.zeros((9 * N, 2)) 

	# original coordinates
	point_tile[:N] = points

	# upper left 
	point_tile[N:2*N, 0] = points[:,0] - x_length
	point_tile[N:2*N, 1] = points[:,1] + y_length

	# directly above
	point_tile[2*N:3*N, 0] = points[:,0]
	point_tile[2*N:3*N, 1] = points[:,1] + y_length

	# upper right
	point_tile[3*N:4*N, 0] = points[:,0] + x_length
	point_tile[3*N:4*N, 1] = points[:,1] + y_length

	# right
	point_tile[4*N:5*N, 0] = points[:,0] + x_length
	point_tile[4*N:5*N, 1] = points[:,1]

	# lower right
	point_tile[5*N:6*N, 0] = points[:,0] + x_length
	point_tile[5*N:6*N, 1] = points[:,1] - y_length

	# under
	point_tile[6*N:7*N, 0] = points[:,0]  
	point_tile[6*N:7*N, 1] = points[:,1] - y_length

	# lower left
	point_tile[7*N:8*N,0] = points[:,0] - x_length
	point_tile[7*N:8*N,1] = points[:,1] - y_length

	# left 
	point_tile[8*N:,0] = points[:,0] - x_length
	point_tile[8*N:,1] = points[:,1]
	# print('point_tile', point_tile)

	return point_tile
