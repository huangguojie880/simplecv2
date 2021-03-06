import cv2
import numpy as np
from PIL import Image


def resize_image_with_crop_or_pad(img, targe):
    """Crops and/or pads an image to a target width and height.
    Resizes an image to a target width and height by either centrally
    cropping the image or padding it evenly with zeros.
    If `width` or `height` is greater than the specified `target_width` or
    `target_height` respectively, this op centrally crops along that dimension.
    If `width` or `height` is smaller than the specified `target_width` or
    `target_height` respectively, this op centrally pads with 0 along that
    dimension.
    Args:
      img: 3-D  shape `[height, width, channels]`.
      targe: tuple `(targe_width,targe_height)`.
    Returns:
      A numpy array, img after resize.
      a 3-D uint8 shape.
    [new_height, new_width, channels]`.
    """
    targe_w = np.int(targe[0])
    targe_h = np.int(targe[1])
    imgShape_len = len(img.shape)
    if imgShape_len == 2:
        img = np.expand_dims(img,axis=-1)
    h, w, deep_dim = img.shape[0], img.shape[1], img.shape[2]
    w_half = int(np.floor(w / 2))
    h_half = int(np.floor(h / 2))
    img_new = np.zeros(shape=(targe_h, targe_w, deep_dim))
    targe_w_half = int(np.floor(targe_w / 2))
    targe_h_half = int(np.floor(targe_h / 2))
    if h > targe_h:
        targe_h_s = 0
        targe_h_e = targe_h + 1
        h_s = h_half - targe_h_half
        h_e = h_half - targe_h_half + targe_h
    else:
        targe_h_s = targe_h_half - h_half
        targe_h_e = targe_h_half - h_half + h
        h_s = 0
        h_e = h + 1
    if w > targe_w:
        targe_w_s = 0
        targe_w_e = targe_w + 1
        w_s = w_half - targe_w_half
        w_e = w_half - targe_w_half + targe_w
    else:
        targe_w_s = targe_w_half - w_half
        targe_w_e = targe_w_half - w_half + w
        w_s = 0
        w_e = w + 1
    img_new[targe_h_s:targe_h_e, targe_w_s:targe_w_e, :] = (img[h_s:h_e, w_s:w_e, :])
    if imgShape_len == 2:
        img_new = img_new[:,:,0]
    return img_new


def imcrop(img, box):
    """Crop a picture by specifying the coordinates of the
     top-left point and the coordinates of the bottom-right point.
     Horizontal x axis and vertical y axis.
    Args:
      img: 2D or 3D
      box: tuple(x1,y1,x2,y2):(x1,y1)is top_left_point,(x2,y2)is right_lower_point
    Returns:
      A numpy array, img after crop.
    `[new_height, new_width, channels]`.
    """
    img_shape = img.shape
    box = np.maximum(box,0)
    box[0] = np.minimum(box[0],img_shape[1])
    box[1] = np.minimum(box[1], img_shape[0])
    box[2] = np.minimum(box[2], img_shape[1])
    box[3] = np.minimum(box[3], img_shape[0])
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])
    if x1 > x2 or y1 > y2:
        raise (
        'Error:The coordinates of the upper left point are not less than the coordinates of the lower right point')
    if len(img_shape) == 3:
        img_new = img[y1:y2, x1:x2, :]
    elif len(img_shape) == 2:
        img_new = img[y1:y2, x1:x2]
    else:
        raise ('Error:img must 2D or 3D')
    return img_new

def imshow(img):
    '''
    Simple image display
    :param img: Enter picture
    :param winname:Box name
    :return:None
    '''
    cv2.imshow('img', np.uint8(img))
    cv2.waitKey()


def imrotate(img, angle):
	'''
	rotate a image
	:param img: Enter picture
	:param angle: rotate angle
	:return: image after rotate
	'''
	img_new = Image.fromarray(img)
	img_new = img_new.rotate(angle, expand=1)
	img_new = np.array(img_new)
	return img_new

def rgb2gray(img):
	'''
	Turn rgb pictures into gray
	:param img: input image
	:return: gray image
	'''
	gray = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2GRAY)
	return gray

def imedge_canny(img, low=50, high=100):
	'''
	Use canny to get the edge of the image
	:param img: gray or rgb
	:param low:Canny's low threshold
	:param high:Canny's high threshold
	:return:edge img
	'''
	img_shape = img.shape
	if len(img_shape) == 3:
		gray = rgb2gray(img)
	else:
		gray = img
	canny = cv2.Canny(gray, low, high)
	return canny


def imextremum(img, type='max'):
	'''
	Find the maximum or minimum value of the picture
	:param img: -
	:param type: 'max' or 'min'
	:return:extremum
	'''
	if type == 'max':
		value = np.max(img)
	elif type == 'min':
		value = np.min(img)
	else:
		raise ('Error:Type must be one of max or min')
	return value

def mergeedge(img, edge):
	'''
	Merge img and edge pictures.
	:param img:Original image,2D or 3D
	:param edge:Edge image,2D
	:return:Merge image
	'''
	img_shape = img.shape
	edge_shape = edge.shape
	if img_shape[0] != edge_shape[0] or img_shape[1] != edge_shape[1]:
		raise ('Error:pictures img and edge must be the same size')
	if len(edge_shape) != 2:
		raise ('Error:edge must 2D')
	if len(img_shape) == 3:
		img[:, :, 0] = np.minimum(img[:, :, 0] + edge, 255)
		img[:, :, 1] = np.minimum(img[:, :, 1] + edge, 255)
		img[:, :, 2] = np.minimum(img[:, :, 2] + edge, 255)
	elif len(img_shape) == 2:
		img = np.minimum(img + edge, 255)
	else:
		raise ('Error:img must 2D or 3D')
	return img
