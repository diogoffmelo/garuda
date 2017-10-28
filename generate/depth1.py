import random


import PIL
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

MODE = '1'
BKG = 1
SIZE = (120, 40)

try:
	FONT = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 25)
except Exception:
	FONT = None

_Y = 6
_XS = list(range(10, 110, 20))


def generate_tiks(length, tiks, lpad=0.1, rpad=0.1):
	l = int(length*lpad)
	r = int(length*(1-rpad))
	delta = int(length/tiks)
	return range(l, r, delta)

def choices(population, k):
	return (random.choice(population) for _ in range(k))


generate_code = choices

def new_image():
	image = Image.new(mode=MODE, size=SIZE, color=1)
	draw = ImageDraw.Draw(image)
	return image, draw

def draw_code(code, draw, tiks):
	for (c, x) in zip(code, tiks):
		if c != '-':
			draw.text(xy=(x,_Y), text=c, font=FONT)


def generate_image(alphabet, k, **kwargs):
	alphabet = list(alphabet)

	if kwargs.get('blank', False):
		 alphabet.append('-')

	code = generate_code(alphabet, k)
	image, draw = new_image()
	
	tiks = generate_tiks(120, k, **kwargs)
	
	draw_code(code, draw, tiks)
	return image, code
	

def generate_images(alphabet, k, N, **kwargs):
	return (generate_image(alphabet, k, **kwargs) for _ in range(N))

