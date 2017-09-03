# the example script to extract features using pyTorch CNN model
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import pdb
import numpy as np
from scipy.misc import imresize as imresize
import cv2
from PIL import Image
from dataset import Dataset
import torch.utils.data as data

# image datasest to be processed
name_dataset = 'testset1';
root_images = 'data_features';
imglist = textread(fullfile(root_images, 'imagelist.txt'),'%s');
num_images = numel(imageList);
for i=1:num_images
    imglist{i} = fullfile(root_images, imglist{i});
end



# load the pre-trained weights
name_model = 'wideresnet_places365'
model_file = '/data/vision/oliva/scenedataset/places2new/models/whole_wideresnet18_places365.pth.tar'
model = torch.load(model_file)
model.eval()
model.cuda()

features_names = ['avgpool']
#features_names = ['layer4','avgpool'] # this is the last conv layer and global average pooling layers



features_blobs = []
def hook_feature(module, input, output):
    # hook the feature extractor
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

for name in features_names:
    model._modules.get(name).register_forward_hook(hook_feature)

# dataset setup
img_size = (224, 224) # input image size
batch_size = 64
num_workers = 6

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

# save variables
imglist_results = []
features_results = [None] * len(features_names)
num_batches = len(dataset) / batch_size
for batch_idx, (input, paths) in enumerate(loader):
    del features_blobs[:]
    print '%d / %d' % (batch_idx, num_batches)
    input = input.cuda()
    input_var = V(input, volatile=True)
    logit = model.forward(input_var)
    imglist_results = imglist_results + list(paths)
    if features_results[0] is None:
        # initialize the feature variable
        for i, feat_batch in enumerate(features_blobs):
            size_features = ()
            size_features = size_features + (len(dataset),)
            size_features = size_features + feat_batch.shape[1:]
            features_results[i] = np.zeros(size_features)
            print features_results[i].shape
    start_idx = batch_idx*batch_size
    end_idx = min((batch_idx+1)*batch_size, len(dataset))
    for i, feat_batch in enumerate(features_blobs):
        features_results[i][start_idx:end_idx] = feat_batch

# save the features
save_name = name_dataset  + '_' + name_model
np.savez('%s.npz'%save_name, features=features_results, imglist=imglist, features_names=features_names)

save_matlab = 0
if save_matlab == 1:
    import scipy.io
    scipy.io.savemat('%s.mat'%save_name, mdict={'list': imglist_results, 'features': features_results[0]})
