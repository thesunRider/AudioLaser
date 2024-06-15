from progress.bar import Bar
import numpy as np
from scipy.special import jv, jn_zeros
from functools import lru_cache
from scipy import spatial
import time,os
import open3d as o3d
from datetime import datetime
from snnpy import *
import  pygame,math,distinctipy
from pygame.locals import*
import neat

#trying to implement NN from https://github.com/maxontech/DriveAI/tree/master

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


enable_screen_vis = True
enable_3d_vis = False
RUN_FRAMES = 2250

jv_array_table = []
jv_array = np.array([])

RADIUS = 1
SIZE_STEP = 100
NEIGBR_COUNT = 10
SPEED_OF_SOUND = 0.75
BESSEL_ROOTS = [jn_zeros(m, 10) for m in range(10)]
FPS = 120
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

#mode_amps = [1,1,1, 1,0,0, 0,0,0]


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

distance_from_origin = 100
screen_width = 800
screen_height = 650
screen_cord = np.array([ [screen_height,screen_width,distance_from_origin],
				[screen_height,-screen_width,distance_from_origin],
				[-screen_height,-screen_width,distance_from_origin],
				[-screen_height,screen_width,distance_from_origin]])



#visualisation_setup 
if enable_3d_vis:
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

if enable_screen_vis:
	screen_color = (49, 150, 100)
	line_color = (255, 0, 0)

	screen = pygame.display.set_mode((screen_width,screen_height))
	screen.fill(screen_color)
	pygame.draw.line(screen,line_color, (60, 80), (130, 100))
	pygame.display.flip()

trck_scrn = pygame.image.load("follow_love.png")
imagerect = trck_scrn.get_rect()
if enable_screen_vis:
	screen.blit(trck_scrn, imagerect)
	pygame.display.flip()



def update(t,mode_amps,class_color): 
	#bar.next()
	#t = i / FPS
	#m, n = MODES[int(t // TIME_PER_MODE)]

	z = np.zeros_like(x)
	for i in range(0,len(MODES)):
		cur_mode = MODES[i]
		#print(mode_amps,i)
		goman = mode_amps[i]
		z += mode_amps[i] * circular_membrane(r, theta, t, cur_mode[0], cur_mode[1], RADIUS, SPEED_OF_SOUND)
	
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
		intersection_point = np.array(P0 + t * d)
		return (intersection_point,True)
	

	closest_point_indx = findIntersection()
	find_close_points_index = getNearbypoints(closest_point_indx,0.1)
	normal_vctr = getNormalvctr(xyz[find_close_points_index])
	point_intersect = xyz[closest_point_indx]

	incident_line = line_origin - point_intersect 
	incident_vctr = incident_line / np.sqrt(np.sum(incident_line**2))
	reflected_vtcr = -incident_vctr - 2 * np.dot(-incident_vctr, normal_vctr) * normal_vctr
	unit_reflected = reflected_vtcr / np.sqrt(np.sum(reflected_vtcr**2))

	point_on_screen = line_plane_intersection(point_intersect,unit_reflected,screen_cord[0],np.array([0,0,-1]))


	##visualisation
	if enable_3d_vis:
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

class rayproj(object):
	"""Class for ray projected"""
	def __init__(self,scrn_input,time_step, color):
		super(rayproj, self).__init__()
		self.color = color
		self.t = 0
		self.cur_pos = np.array([0,0])
		self.last_pos = np.array([0,0])
		self.screen = scrn_input[1]
		self.trck_scrn = scrn_input[0]
		self.trck_screen_dim = scrn_input[0].get_rect()
		self.time_step = time_step
		self.angle = 0
		self.radars = []
		self.offset_location = np.array([200,-60])
		self.distance = 0

	def hit(self,pos,box):
		return box[0]+box[2] > pos[0] > box[0] and box[1]+box[3] > pos[1] > box[1]

	def check_collision(self,radius):
		#check a semicircle infront of the ray at a radius if it has went out of track
		if self.hit(self.cur_pos,[0,0,self.trck_screen_dim.w,self.trck_screen_dim.h]):
			for i in range(0,180,20):
				cur_color = self.trck_scrn.get_at((int(self.cur_pos[0]+radius*math.sin(math.radians(i))),int(self.cur_pos[1]+radius*math.cos(math.radians(i)))))
				if not cur_color == pygame.Color(255,255, 255, 255):
					return True

			#no collisions so far on path
			return False

		else:
			return True

	def radar(self, radar_angle):
		length = 0
		x = int(self.cur_pos[0])
		y = int(self.cur_pos[1])

		if self.hit(self.cur_pos,[0,0,self.trck_screen_dim.w,self.trck_screen_dim.h]):
			while self.trck_scrn.get_at((x, y)) == pygame.Color(255,255, 255, 255) and length < 100:
				length += 1
				x = int(self.cur_pos[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
				y = int(self.cur_pos[1] - math.sin(math.radians(self.angle + radar_angle)) * length)
				if not self.hit(np.array([x,y]),[0,0,self.trck_screen_dim.w,self.trck_screen_dim.h]):
					break;

		# Draw Radar
		pygame.draw.line(self.screen, (255,0 , 0, 255), self.cur_pos, (x, y), 1)

		dist = int(math.sqrt(math.pow(self.cur_pos[0] - x, 2)
							 + math.pow(self.cur_pos[1] - y, 2)))
		self.radars.append([radar_angle, dist])

		
	def get_data(self):
		#5 feelers, current pos
		feelers = np.array([0,0,0,0,0])
		for i, radar in enumerate(self.radars):
			feelers[i] = int(radar[1])

		return feelers

	def update(self,mode_amps):
		#print("hopdating",time.time())
		self.t += self.time_step
		self.radars.clear()
		point_on_screen = update(self.t,mode_amps,self.color)
		if point_on_screen[1]:
			self.cur_pos = (point_on_screen[0])[0:2] + np.array([screen_width/2,screen_height/2]) + self.offset_location #offset cordinates to centre of screen
			pygame.draw.line(screen,self.color, self.last_pos, self.cur_pos)
			pygame.display.flip()
			self.distance += np.linalg.norm(self.cur_pos-self.last_pos)

			self.last_pos = self.cur_pos

			pygame.draw.circle(self.screen, self.color, (int(self.cur_pos[0]),int(self.cur_pos[1])), 3, 1)
			
			for radar_angle in (0, 72, 72*2, 72*3, 72*4):
				self.radar(radar_angle)

			
			pygame.display.flip()

		

rays = []
ge = []
nets = []

def remove_ray(index):
	rays.pop(index)
	ge.pop(index)
	nets.pop(index)

def eval_genomes(genomes, config):
	global rays, ge, nets

	time_step = 1/(10*FPS) #each fps will have 10 steps,increase for more resolution
	check_radius_path = 2 #radius to search for collision

	output_colors = distinctipy.get_colors(len(genomes), [(1,1,1)]) # generate colrs apart from white
	normalised_colors = (np.array([np.array(j) for j in output_colors])*255).astype(int) #color of each competetor

	screen.fill(screen_color)
	screen.blit(trck_scrn, imagerect)	#redraw the screen with custom path,removing all lines and graphics just the path
	pygame.display.flip()
	pygame.display.update()

	k = 0
	for genome_id, genome in genomes:
		rays.append(rayproj((trck_scrn,screen),time_step,normalised_colors[k]))
		ge.append(genome)
		net = neat.nn.FeedForwardNetwork.create(genome, config)
		nets.append(net)
		genome.fitness = 0
		k += 1

	run = True
	while run:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				exit()

		if len(rays) == 0:
			break

		for i, ray_active in enumerate(rays):
			if  ray_active.check_collision(check_radius_path):
				if ray_active.distance < 300:
					ge[i].fitness = 0
				remove_ray(i)
			else:
				if ray_active.t < 10/FPS:
					ge[i].fitness += 0.5 + ray_active.distance/1000 #2/(1+np.sum(ray_active.get_data())/5 ) #not went out of path, good ray increase fitness
				else:
					ge[i].fitness = 0
					remove_ray(i)

		for i, ray_active in enumerate(rays):
			output = nets[i].activate(ray_active.get_data())
			#print("neural out=",output)
			ray_active.update(np.round(output[:len(MODES)],2))



startTime = datetime.now()
#eval_genomes()

# Setup NEAT Neural Network
def run(config_path):
	global pop
	number_gens = 10000

	config = neat.config.Config(
		neat.DefaultGenome,
		neat.DefaultReproduction,
		neat.DefaultSpeciesSet,
		neat.DefaultStagnation,
		config_path
	)

	pop = neat.Population(config)
	pop.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	pop.add_reporter(stats)

	pop.run(eval_genomes, number_gens)



local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config.txt')
run(config_path)
exit()

debug_ray = rayproj((trck_scrn,screen),1/100,(0,255,0))
debug_ray.check_collision(3)
for i in range(0,10):
	debug_ray.update([1,0,0, 0,0,0, 0,0,0])
debug_ray.get_data()
input("")
#for i in range(0,RUN_FRAMES): #2250
#	update(i/20,mode_amps)
#	if enable_3d_vis:
#		time.sleep(1/FPS)
#
#	if enable_screen_vis:
#		for events in pygame.event.get():
#			if events.type == QUIT:
#				exit()

bar.finish()
print("ELT:",datetime.now() - startTime)

if enable_3d_vis:
	vis.destroy_window()