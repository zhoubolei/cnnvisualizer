# the example script to generate the unit segmentation visualization using pyTorch
# Bolei Zhou

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import pdb
import numpy as np
import cv2
from PIL import Image
from dataset import Dataset
import torch.utils.data as data
import torchvision.models as models

# visualization setup
img_size = (224, 224)       # input image size
segment_size = (120,120)    # the unit segmentaiton size
num_top = 12                # how many top activated images to extract
margin = 3                  # pixels between two segments
threshold_scale = 0.2       # the scale used to segment the feature map. Smaller the segmentation will be tighter.
flag_crop = 0               # whether to generate tight crop for the unit visualiation.
flag_classspecific = 1      # whether to generate the class specific unit for each category (only works for network with global average pooling at the end)

# dataset setup
batch_size = 64
num_workers = 6


"""
using old version of pytorch to load new torch models
"""
 # hacky way to deal with the Pytorch 1.0 update
def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module

def load_model():
    # this model has a last conv feature map as 14x14

    model_file = 'wideresnet18_places365.pth.tar'
    if not os.access(model_file, os.W_OK):
        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
        os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')

    import wideresnet
    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
    for i, (name, module) in enumerate(model._modules.items()):
        module = recursion_change_bn(model)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)
    model.eval()
    model.cuda()
    return model

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


# load model
model = load_model()
print(model)
model_name = 'wideresnet_places365'


# config the class list file here
class_file = 'categories_places365.txt'
if not os.path.exists(class_file):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    os.system('wget ' + synset_url)

if not os.path.exists(class_file):
    print('Your category list does not exist')
    raise FileNotFoundError
classes = list()
with open(class_file) as f:
    for line in f:
            classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)
# feature extraction layer setup
features_names = ['layer4']


# image datasest to be processed
name_dataset = 'sun+imagenetval'
root_image = 'images'
with open('images/imagelist.txt') as f:
    lines = f.readlines()
imglist = []
for line in lines:
    line = line.rstrip()
    imglist.append(root_image + '/' + line)

features_blobs = []
def hook_feature(module, input, output):
    # hook the feature extractor
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

for name in features_names:
    model._modules.get(name).register_forward_hook(hook_feature)

# image transformer
tf = trn.Compose([
        trn.Scale(img_size),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = Dataset(imglist, tf)
loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False)

# extract the max value activaiton for each image
imglist_results = []
maxfeatures = [None] * len(features_names)
num_batches = len(dataset) / batch_size
for batch_idx, (input, paths) in enumerate(loader):
    del features_blobs[:]
    print('%d / %d' % (batch_idx+1, num_batches))
    input = input.cuda()
    input_var = V(input, volatile=True)
    logit = model.forward(input_var)
    imglist_results = imglist_results + list(paths)
    if maxfeatures[0] is None:
        # initialize the feature variable
        for i, feat_batch in enumerate(features_blobs):
            size_features = (len(dataset), feat_batch.shape[1])
            maxfeatures[i] = np.zeros(size_features)
    start_idx = batch_idx*batch_size
    end_idx = min((batch_idx+1)*batch_size, len(dataset))
    for i, feat_batch in enumerate(features_blobs):
        maxfeatures[i][start_idx:end_idx] = np.max(np.max(feat_batch,3),2)

# generate the top activated images
output_folder = 'result_segments/%s' % model_name
if not os.path.exists(output_folder):
    os.makedirs(output_folder + '/image')

# output the html first
for layerID, layer in enumerate(features_names):
    file_html = os.path.join(output_folder, layer + '.html')
    with open(file_html, 'w') as f:
        num_units = maxfeatures[layerID].shape[1]
        lines_units = ['%s-unit%03d.jpg' % (layer, unitID) for unitID in range(num_units)]
        lines_units = ['unit%03d<br><img src="image/%s">'%(unitID, lines_units[unitID]) for unitID in range(num_units)]
        f.write('\n<br>'.join(lines_units))

    # it contains the cropped regions
    if flag_crop == 1:
        file_html_crop = os.path.join(output_folder, layer + '_crop.html')
        with open(file_html_crop, 'w') as f:
            num_units = maxfeatures[layerID].shape[1]
            lines_units = ['%s-unit%03d_crop.jpg' % (layer, unitID) for unitID in range(num_units)]
            lines_units = ['unit%03d<br><img src="image/%s">'%(unitID, lines_units[unitID]) for unitID in range(num_units)]
            f.write('\n<br>'.join(lines_units))

if flag_classspecific == 1:
    num_topunit_class = 3
    layer_lastconv = features_names[-1]
    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    file_html = os.path.join(output_folder, 'class_specific_unit.html')
    output_lines = []
    for classID in range(len(classes)):
        line = '<h2>%s</h2>' % classes[classID]
        idx_units_sorted = np.argsort(np.squeeze(weight_softmax[classID]))[::-1]
        for sortID in range(num_topunit_class):
            unitID = idx_units_sorted[sortID]
            weight_unit = weight_softmax[classID][unitID]
            line += 'weight=%.3f %s<br>' % (weight_unit, lines_units[unitID])
        line = '<p>%s</p>' % line
        output_lines.append(line)

    with open(file_html, 'w') as f:
        f.write('\n'.join(output_lines))


# generate the unit visualization
for layerID, layer in enumerate(features_names):
    num_units = maxfeatures[layerID].shape[1]
    imglist_sorted = []
    # load the top actiatied image list into one list
    for unitID in range(num_units):
        activations_unit = np.squeeze(maxfeatures[layerID][:, unitID])
        idx_sorted = np.argsort(activations_unit)[::-1]
        imglist_sorted += [imglist[item] for item in idx_sorted[:num_top]]

    # data loader for the top activated images
    loader_top = data.DataLoader(
        Dataset(imglist_sorted, tf),
        batch_size=num_top,
        num_workers=num_workers,
        shuffle=False)
    for unitID, (input, paths) in enumerate(loader_top):
        del features_blobs[:]
        print('%d / %d' % (unitID+1, num_units))
        input = input.cuda()
        input_var = V(input, volatile=True)
        logit = model.forward(input_var)
        feature_maps = features_blobs[layerID]

        images_input = input.cpu().numpy()
        max_value = 0
        output_unit = []
        for i in range(num_top):
            feature_map = feature_maps[i][unitID]
            if max_value == 0:
                max_value = np.max(feature_map)
            feature_map = feature_map / max_value
            mask = cv2.resize(feature_map, segment_size)
            mask[mask < threshold_scale] = 0.0 # binarize the mask
            mask[mask > threshold_scale] = 1.0

            img = cv2.imread(paths[i])
            img = cv2.resize(img, segment_size)
            img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
            img_mask = np.multiply(img, mask[:,:, np.newaxis])
            img_mask = np.uint8(img_mask * 255)
            output_unit.append(img_mask)
            output_unit.append(np.uint8(np.ones((segment_size[0],margin,3))*255))
        montage_unit = np.concatenate(output_unit, axis=1)
        cv2.imwrite(os.path.join(output_folder, 'image', '%s-unit%03d.jpg'%(layer, unitID)), montage_unit)
        if flag_crop == 1:
            # load the library to crop image
            import tightcrop
            montage_unit_crop = tightcrop.crop_tiled_image(montage_unit, margin)
            cv2.imwrite(os.path.join(output_folder, 'image', '%s-unit%03d_crop.jpg'%(layer, unitID)), montage_unit_crop)
print('done check results in ' + output_folder)
