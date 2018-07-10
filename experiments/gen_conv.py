


cross1 = np.array([[0,0,0],[1,1,1],[0,0,0]])/3.0
cross2 = np.array([[0,1,0],[0,1,0],[0,1,0]])/3.0

#cross1 = np.array([[-1,-1,-1],[1,1,1],[-1,-1,1]])/3.0
#cross2 = np.array([[0,1,0],[0,1,0],[0,1,0]])/3.0



def crosses(size=(10, 10), n=3):
	blanck_image = np.zeros(size)
	#for _ in range(n):
	#	x = np.random.choice(np.arange(1, size[0]-1))
	#	y = np.random.choice(np.arange(1, size[1]-1))
	#	blanck_image[x-1:x+2, y-1:y+2][cross1 > 0] = 1

	x3 = int(size[0]/3)
	y3 = int(size[0]/3)

	blanck_image[x3, :] = 1
	blanck_image[2*x3, :] = 1

	blanck_image[:, y3] = 1
	blanck_image[:, 2*y3] = 1


	#blanck_image[x3, y3:2*y3] = 1
	#blanck_image[2*x3, y3:2*y3] = 1

	#blanck_image[x3:2*x3, y3] = 1
	#blanck_image[x3:2*x3, 2*y3] = 1

	return blanck_image

def conv(image, kernel):
	return ndimage.convolve(image, kernel, mode='constant', cval=0.0)


def sample_random_image():
	cmap=plt.get_cmap('gist_yarg')

	#img = ndimage.gaussian_filter(crosses(), sigma=0.5)
	img = crosses(size=(20, 20))

	#ref = np.max(img)
	kimg = img.copy()
	#kimg[0, 0] = 1

	plt.figure(0)
	plt.imshow(kimg, cmap=cmap)

	plt.figure(1)
	kimg = conv(img, cross1)
	
	kimg[kimg < 0.2] = 0
	#kimg[kimg >= 0.2] = 1

	#kimg[0, 0] = 1
	#kimg = kimg/np.max(kimg)
	plt.imshow(kimg, cmap=cmap)
	#plt.imshow(cross1, cmap=cmap)


	plt.figure(2)
	kimg = conv(img, cross2)
	kimg[kimg < 0.2] = 0
	#kimg[kimg >= 0.2] = 1
	
	#kimg[0, 0] = 1
	#kimg = kimg/np.max(kimg)
	plt.imshow(kimg, cmap=cmap)

sample_random_image()