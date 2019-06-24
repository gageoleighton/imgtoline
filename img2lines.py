import PIL, os
from PIL import Image
import tkinter as tk
root = tk.Tk()
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage import img_as_float
import matplotlib.pyplot as plt
import datetime, time, math
from random import *
import random
import networkx as nx

root.withdraw()
root.update()
imagename = askopenfilename()
time.sleep(1)
root.destroy()
time.sleep(1)

image = cv2.imread(imagename)
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#grayscale_image.show()

height, width = grayscale_image.shape
print(height, width) # 326 x 299

#initialize white image
white_image = np.zeros((height,width), np.uint8)
white_image[:,:] = (255)

def compare_images(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype(float) - imageB.astype(float)) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def sim(imageA, imageB):
	# compute the mean squared error and structural similarity
	# index for the images
	s = ssim(imageA, imageB)
	return s

def distance_from_line(x0,y0,x1,y1,x2,y2):
	if y2 == y1 and x2 == x1:
		return 0
	else:
		return math.fabs(((y2-y1)*x0-(x2-x1)*y0+x2*y1-y2*x1)/math.sqrt(((y2-y1)**2)+(x2-x1)**2))

def getboxpoints():
	p = random.randint(0, width + width + height + height)
	if p < (width + height):
		if p < width:
			x = p
			y = 0
		else:
			x = width
			y = p - width
	else:
		p2 = p - (width + height)
		if p2 < width:
			x = width - p2
			y = height
		else:
			x = 0
			y = height - (p2 - width)
	return [x,y,p]

def longest_path(G):
    dist = {} # stores [node, distance] pair
    for node in nx.topological_sort(G):
        # pairs of dist,node for all incoming edges
        pairs = [(dist[v][0]+1,v) for v in G.pred[node]] 
        if pairs:
            dist[node] = max(pairs)
        else:
            dist[node] = (0, node)
    node,(length,_)  = max(dist.items(), key=lambda x:x[1])
    path = []
    while length > 0:
        path.append(node)
        length,node = dist[node]
    return list(reversed(path))

path = []
starting_point = 0
ending_point = 0

final_image_edit = white_image.copy()
starting_time = time.clock()
total_mes_per_sec = 0
last_image_edit = final_image_edit.copy()
new_s = sim(grayscale_image,last_image_edit)


darkness = 4 # Adjust the darkness of each line
away = 1 # Must be less than darkness
false_counts = 0
total = 0

g1 = nx.DiGraph()

print("step 1")
while (total <= width*height/(darkness + away)): # You can edit these to obtain more lines per image

	last_image_edit = final_image_edit.copy()
	old_s = sim(grayscale_image,last_image_edit)

	[x1,y1,p1] = getboxpoints()
	[x2,y2,p2] = getboxpoints()

	if (math.sqrt((x1-x2)**2 + (y1-y2)**2) >= 10):
		image_edit = last_image_edit.copy()

		for x3 in range(0,height):
			for y3 in range(0,width):
				distance = distance_from_line(x1,y1,x2,y2,x3,y3)
				if (distance > 0.25) and (distance <= away): # Make this resemble actual line
					if image_edit[x3,y3] >= math.ceil(darkness/distance):
						image_edit[x3,y3] -= math.ceil(darkness/distance)
					else:
						image_edit[x3,y3] = 0
				elif (distance <= 0.25) and (distance >= 0):
					if image_edit[x3,y3] >= darkness*8:
						image_edit[x3,y3] -= darkness*8
					else:
						image_edit[x3,y3] = 0

		new_s = sim(grayscale_image,image_edit)

		if (new_s > old_s): # Must be favorable
			starting_point = p1
			ending_point = p2

			old_s = new_s
			final_image_edit = image_edit.copy()

			path.append([starting_point,ending_point])

			g1.add_node(p1)
			g1.add_node(p2)
			g1.add_edge(p1, p2)

			total += 1
			# f= open(imagename+".txt","w+") # Use this to output the points to a file after each line
			# f.write(str(path))
			# f.close()
			# print((time.clock() - starting_time)/len(path))
			string = str(len(path))+":	"+str(starting_point)
			if(total % 100 == 0): # Will refreash the image every 100 lines
				cv2.destroyAllWindows()
				cv2.imshow(string, final_image_edit)
				cv2.waitKey(20)

# lpath = 0
# path2 = []
# for x in np.unique(path):
# 	for y in np.unique(path):
# 		p = nx.all_simple_paths(g1, x, y, cutoff=(lpath+1))
# 		for a in p:
# 			if len(a) > lpath:
# 				print("New longest path!", len(a), x, y)
# 				lpath = len(a)
# 				path2 = a

# print(list(path2))

#print(path)
print((time.clock() - starting_time)/len(path))
