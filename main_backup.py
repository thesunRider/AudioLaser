from progress.bar import Bar
import numpy as np
from scipy.special import jv, jn_zeros
from functools import lru_cache
from scipy import spatial
import time
import open3d as o3d
from datetime import datetime
from snnpy import *

enable_vis = False
RUN_FRAMES = 2250

jv_array_table = []
jv_array = np.array([])

RADIUS = 1
SIZE_STEP = 100
NEIGBR_COUNT = 10
SPEED_OF_SOUND = 0.75
BESSEL_ROOTS = [jn_zeros(m, 10) for m in range(10)]
FPS = 60
TIME_PER_MODE = 10

MODES = (
	(0, 1),
	(0, 2),
	(0, 3),
	(1, 1),
	(1, 2),
	(1, 3),
	(2, 1),
	(2, 2),
	(2, 3)
)

FRAMES = len(MODES) * TIME_PER_MODE * FPS


def get_jv(b,a):
	return jv(a,b)

@lru_cache(maxsize=None)
def lambda_mn(m, n, radius):
	return BESSEL_ROOTS[m][n - 1] / radius

def circular_membrane(r, theta, t, m, n, radius, speed_of_sound):
	global jv_array
	l = lambda_mn(m, n, radius)
	T = np.sin(speed_of_sound * l * t)

	if (m,n) in jv_array_table:
		R = jv_array[jv_array_table.index((m,n))]
	else:
		jv_array_table.append((m,n))
		R =  get_jv( l * r,m)
		if(len(jv_array) == 0):
			jv_array = np.array([R])
		else:
			jv_array = np.append(jv_array,np.array([R]),axis=0)
		R = jv_array[-1]

	Theta = np.cos(m * theta)

	return R * T * Theta


r = np.linspace(0, RADIUS, SIZE_STEP)
theta = np.linspace(0, 2 * np.pi, SIZE_STEP)

m, n = MODES[0]
r, theta = np.meshgrid(r, theta)
x = np.cos(theta) * r
y = np.sin(theta) * r
z = np.zeros_like(x) #circular_membrane(r, theta, 0, m, n, RADIUS, SPEED_OF_SOUND)
omega = SPEED_OF_SOUND * lambda_mn(m, n, RADIUS)


line_origin = np.array([1,1,1]) #laser begins here
line_end = np.array([-1,-1,-1]) #laser pointed here

reflected_origin = np.array([0,0,0]) # reflectd begins
reflected_end = np.array([1,0,0]) #reflected line end

xyz = np.zeros((np.size(x), 3))
xyz[:, 0] = np.reshape(x, -1)
xyz[:, 1] = np.reshape(y, -1)
xyz[:, 2] = np.reshape(z, -1)

distance_from_origin = 2
screen_width = 1
screen_height = 1
screen_cord = np.array([ [screen_height,screen_width,distance_from_origin],
				[screen_height,-screen_width,distance_from_origin],
				[-screen_height,-screen_width,distance_from_origin],
				[-screen_height,screen_width,distance_from_origin]])



#visualisation_setup 
if enable_vis:
	vis = o3d.visualization.Visualizer()
	vis.create_window()
	opt = vis.get_render_option()
	opt.point_show_normal = False 
	opt.line_width = 5

	# Convert the NumPy array to an Open3D point cloud
	pcd = o3d.geometry.PointCloud()

	pcd.points = o3d.utility.Vector3dVector(xyz)
	normals = np.full_like(np.asarray(pcd.points),[0,0,1])
	pcd.normals = o3d.utility.Vector3dVector(normals)
	pcd.paint_uniform_color([0,0,1])

	line_set = o3d.geometry.LineSet()
	line_set.points = o3d.utility.Vector3dVector([line_origin, line_end,[0,0,0],[0,0,0],reflected_origin,reflected_end])
	line_set.lines = o3d.utility.Vector2iVector([[0, 1],[2,3],[4,5]])
	line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0],[0,0,1],[0,1,0]])
	#green reflected, blue normal , red incident

	screen_line = o3d.geometry.LineSet()
	screen_line.points = o3d.utility.Vector3dVector([screen_cord[0],screen_cord[1] , screen_cord[1],screen_cord[2], screen_cord[2],screen_cord[3], screen_cord[3],screen_cord[0]])
	screen_line.lines = o3d.utility.Vector2iVector([[0, 1],[2,3],[4,5],[6,7]])

	vis.add_geometry(line_set)
	vis.add_geometry(pcd)
	vis.add_geometry(screen_line)
	vis.poll_events()
	vis.update_renderer()

	#vis.run() #needed for camera
	#vis.destroy_window()
	#exit()



bar = Bar('Processing', max=RUN_FRAMES, suffix='%(percent)d%% [%(avg)s / %(eta)d ]')

def update(i): 
	bar.next()
	t = i / FPS
	m, n = MODES[int(t // TIME_PER_MODE)]

	z = circular_membrane(r, theta, t, m, n, RADIUS, SPEED_OF_SOUND)
	omega = SPEED_OF_SOUND * lambda_mn(m, n, RADIUS)

	xyz[:, 2] = np.reshape(z, -1)

	def getNearbypoints(indx,radius):
		snn_model = build_snn_model(xyz)  
		found = snn_model.query_radius(xyz[indx], radius)
		return found

	def findIntersection():
		product = np.cross(xyz - line_origin, line_end - line_origin)
		if product.ndim == 2:
			distances = np.linalg.norm(product, axis=1)
		else:
			distances = np.abs(product)
		return distances.argmin()

	def getNormalvctr(pnts):
		centroid = np.mean(pnts, axis=0)
		centered_points = pnts - centroid
		covariance_matrix = np.cov(centered_points, rowvar=False)
		eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
		normal = eigenvectors[:, np.argmin(eigenvalues)]
		if np.dot(normal,np.array([0,0,1]) ) <0:
			normal = - normal
		return normal

	def line_plane_intersection(P0, d, P1, n):
		dot_nd = np.dot(n, d)
		if np.isclose(dot_nd, 0):
			return ([],False)

		t = np.dot(n, (P1 - P0)) / dot_nd
		intersection_point = P0 + t * d
		return (intersection_point,True)
	

	closest_point_indx = findIntersection()
	find_close_points_index = getNearbypoints(closest_point_indx,0.05)
	normal_vctr = getNormalvctr(xyz[find_close_points_index])
	point_intersect = xyz[closest_point_indx]

	incident_line = line_origin - point_intersect 
	incident_vctr = incident_line / np.sqrt(np.sum(incident_line**2))
	reflected_vtcr = -incident_vctr - 2 * np.dot(-incident_vctr, normal_vctr) * normal_vctr
	unit_reflected = reflected_vtcr / np.sqrt(np.sum(reflected_vtcr**2))

	point_on_screen = line_plane_intersection(point_intersect,unit_reflected,screen_cord[0],np.array([0,0,-1]))


	##visualisation
	if enable_vis:
		if point_on_screen[1]:
			ary_apnd = np.append(xyz,[point_on_screen[0]],axis=0)
		else:
			ary_apnd = xyz

		pcd.points = o3d.utility.Vector3dVector(ary_apnd)
		pcd.paint_uniform_color([0,0,1])
		pcd_colors = np.asarray(pcd.colors)
		pcd_colors[find_close_points_index] = [1,0,0]
		pcd.colors =  o3d.utility.Vector3dVector(pcd_colors)
		
		normals = np.full_like(np.asarray(pcd.points),normal_vctr)
		pcd.normals = o3d.utility.Vector3dVector(normals)
		#pcd.estimate_normals()
		line_set.points = o3d.utility.Vector3dVector([line_origin, line_end, point_intersect,point_intersect + normal_vctr,point_intersect ,point_intersect+ unit_reflected*5 ])

		vis.update_geometry(pcd)
		vis.update_geometry(line_set)
		vis.poll_events()
		vis.update_renderer()

	return point_on_screen

startTime = datetime.now()
for i in range(0,RUN_FRAMES): #2250
	update(i)
	if enable_vis:
		time.sleep(1/FPS)	

bar.finish()
print("ELT:",datetime.now() - startTime)

if enable_vis:
	vis.destroy_window()