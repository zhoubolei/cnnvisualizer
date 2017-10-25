
# coding: utf-8

# In[ ]:

# Awesome image patch finder,
# Due to http://stackoverflow.com/questions/9525313

import os
import sys
import numpy
import random

import numpy as np
import scipy.ndimage as ndimage
import scipy.spatial as spatial
import scipy.misc as misc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import urllib2

from scipy.misc import imresize, imread, imsave

class BBox(object):
    def __init__(self, x1, y1, x2, y2):
        '''
        (x1, y1) is the upper left corner,
        (x2, y2) is the lower right corner,
        with (0, 0) being in the upper left corner.
        '''
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    def taxicab_diagonal(self):
        '''
        Return the taxicab distance from (x1,y1) to (x2,y2)
        '''
        return self.x2 - self.x1 + self.y2 - self.y1
    def overlaps(self, other):
        '''
        Return True iff self and other overlap.
        '''
        return not ((self.x1 > other.x2)
                    or (self.x2 < other.x1)
                    or (self.y1 > other.y2)
                    or (self.y2 < other.y1))
    def __eq__(self, other):
        return (self.x1 == other.x1
                and self.y1 == other.y1
                and self.x2 == other.x2
                and self.y2 == other.y2)

def find_paws(data, smooth_radius = 5, threshold = 0.0001):
    # http://stackoverflow.com/questions/4087919/how-can-i-improve-my-paw-detection
    """Detects and isolates contiguous regions in the input array"""
    # Blur the input data a bit so the paws have a continous footprint
    data = ndimage.uniform_filter(data, smooth_radius)
    # Threshold the blurred data (this needs to be a bit > 0 due to the blur)
    thresh = data > threshold
    # Fill any interior holes in the paws to get cleaner regions...
    filled = ndimage.morphology.binary_fill_holes(thresh)
    # Label each contiguous paw
    coded_paws, num_paws = ndimage.label(filled)
    # Isolate the extent of each paw
    # find_objects returns a list of 2-tuples: (slice(...), slice(...))
    # which represents a rectangular box around the object
    data_slices = ndimage.find_objects(coded_paws)
    return data_slices

def slice_to_bbox(slices):
    for s in slices:
        dy, dx = s[:2]
        yield BBox(dx.start, dy.start, dx.stop+1, dy.stop+1)

def remove_overlaps(bboxes):
    '''
    Return a set of BBoxes which contain the given BBoxes.
    When two BBoxes overlap, replace both with the minimal BBox that contains both.
    '''
    # list upper left and lower right corners of the Bboxes
    corners = []

    # list upper left corners of the Bboxes
    ulcorners = []

    # dict mapping corners to Bboxes.
    bbox_map = {}

    for bbox in bboxes:
        ul = (bbox.x1, bbox.y1)
        lr = (bbox.x2, bbox.y2)
        bbox_map[ul] = bbox
        bbox_map[lr] = bbox
        ulcorners.append(ul)
        corners.append(ul)
        corners.append(lr)

    # Use a KDTree so we can find corners that are nearby efficiently.
    tree = spatial.KDTree(corners)
    new_corners = []
    for corner in ulcorners:
        bbox = bbox_map[corner]
        # Find all points which are within a taxicab distance of corner
        indices = tree.query_ball_point(
            corner, bbox_map[corner].taxicab_diagonal(), p = 1)
        for near_corner in tree.data[indices]:
            near_bbox = bbox_map[tuple(near_corner)]
            if bbox != near_bbox and bbox.overlaps(near_bbox):
                # Expand both bboxes.
                # Since we mutate the bbox, all references to this bbox in
                # bbox_map are updated simultaneously.
                bbox.x1 = near_bbox.x1 = min(bbox.x1, near_bbox.x1)
                bbox.y1 = near_bbox.y1 = min(bbox.y1, near_bbox.y1)
                bbox.x2 = near_bbox.x2 = max(bbox.x2, near_bbox.x2)
                bbox.y2 = near_bbox.y2 = max(bbox.y2, near_bbox.y2)
    return set(bbox_map.values())

if False:
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # data = misc.imread(urllib2.urlopen('https://i.stack.imgur.com/ueOHC.png'))
    data = misc.imread('/data/vision/torralba/gigaSUN/www/unit_annotation/result_segments_iterations/caffeNet_imagenet2places_iter_1/html/image/conv5-0000.jpg')
    data = data[:160,3:163,:]

    im = ax.imshow(data)
    data_slices = find_paws(data, smooth_radius = 20, threshold = 52)

    bboxes = remove_overlaps(slice_to_bbox(data_slices))
    for bbox in bboxes:
        xwidth = bbox.x2 - bbox.x1
        ywidth = bbox.y2 - bbox.y1
        p = patches.Rectangle((bbox.x1, bbox.y1), xwidth, ywidth,
                              fc = 'none', ec = 'red')
        ax.add_patch(p)

    plt.show()



def input_image_filename(basename, iter, zunit):
#    return ('/data/vision/torralba/gigaSUN/www/unit_annotation/result_segments_iterations' +
    return ('/data/vision/torralba/scratch2/davidbau/iccv' +
        '/%s_iter_%d/html/image/conv5-%04d.jpg' % (basename, iter, zunit))

def output_image_filename(basename, iter, zunit):
#    return ('/data/vision/torralba/gigaSUN/www/unit_annotation/result_segments_iterations' +
    return ('/data/vision/torralba/scratch2/davidbau/iccv' +
        '/%s_iter_%d/html/image/conv5-%04d_crop.jpg' % (basename, iter, zunit))

def biggest_square_bbox(data, smooth_radius, threshold):
    data_slices = find_paws(data, smooth_radius=smooth_radius, threshold=threshold)
    if not data_slices:
        return BBox(0,0,0,0)
    bboxes = remove_overlaps(slice_to_bbox(data_slices))
    largest = max(bboxes, key=lambda b: (b.x2 - b.x1) * (b.y2 - b.y1))
    bwidth = largest.x2 - largest.x1 - 1
    bheight = largest.y2 - largest.y1 - 1

    if bwidth > bheight:
        # Height needs to expand, but keep it legal
        largest.y1 = min(data.shape[1] - bwidth, max(0,
                (largest.y1 + largest.y2 - bwidth) // 2))
        largest.y2 = largest.y1 + bwidth + 1
    else:
        # Width needs to expand, but keep it legal
        largest.x1 = min(data.shape[0] - bheight, max(0,
                (largest.x1 + largest.x2 - bheight) // 2))
        largest.x2 = largest.x1 + bheight + 1
    return largest

def best_tightcrop(imagedata):
    bbox = biggest_square_bbox(imagedata, smooth_radius = 10, threshold = 50)
    # print bbox.x1, bbox.x2, bbox.y1, bbox.y2
    # Don't zoom in to too tiny an area - just return the original
    if bbox.x2 - bbox.x1 < 10:
        return imagedata
    return imresize(imagedata[bbox.y1:bbox.y2, bbox.x1:bbox.x2, :], imagedata.shape[:2])

def crop_tiled_image(data, border=3):
    outdata = data.copy()
    width = data.shape[0]
    for x in range(0, data.shape[1], width + border):
        onesquare = data[:,x:x+width,:]
        outdata[:,x:x+width,:] = best_tightcrop(onesquare)
    return outdata

def process_iteration(basename, iteration, show_things=False):
    for zunit in range(256):
        input_image = input_image_filename(basename, iteration, zunit)
        output_image = output_image_filename(basename, iteration, zunit)
        if os.path.exists(output_image)==False and os.path.exists(input_image)==True:
            print('processing ' + input_image)
            data = imread(input_image)
            cropped = crop_tiled_image(data)
            imsave(output_image, cropped)
            if show_things:
                plt.imshow(data)
                plt.show()
                plt.imshow(crop_tiled_image(data))
                plt.show()


def process_snapshot():
    name = '/data/vision/oliva/scenedataset/modelzoo/list_iterations.txt'
    with open(name) as f:
        lines = f.readlines()
    for line in lines:
        print(line)
        model_name = line.rstrip()
        items = model_name.split('/')
        model_name = items[1]
        items = model_name.split('.')
        model_name = items[0]
        items = model_name.split('_')
        basename = '_'.join(items[:-2])
        iteration = int(items[-1])
        print(basename, iteration)
        process_iteration(basename, iteration)

def process_snapshot_iterations():
    basename = 'places'
    iters = [1,2,4,9,20,44,99,223,492,1108,2446,5509,12164,27396,60491,136238,300818,600818,1200818,2400818]
    for iteration in iters:
        process_iteration(basename, iteration)


