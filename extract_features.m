% simple script to extract CNN features from any layers of CNN for any given images
% written by Bolei Zhou. 
% June 20, 2017

clear
% the location of the caffe you install

addpath('/yourpah/caffe/matlab'); % make sure you compile the matcaffe

root_features = 'mat_files'; % the location where you want to output the CNN features
root_model = 'model'; % the location where you put the CNN models
netID = 3;

device_id = 0; % GPU ID to use


% load the testing images
data_name = 'talia_june20';
root_images = '/data/vision/torralba/gigaSUN/www/unit_annotation/data_features';
imageList = textread(fullfile(root_images, 'imagelist.txt'),'%s');
num_images = numel(imageList);
for i=1:num_images
    imageList{i} = fullfile(root_images, imageList{i});
end

      
% select the CNN model

if netID == 1
    network = 'caffe_reference_imagenet';
    layers = {'conv5','conv4','conv3','conv2','conv1'};
    batch_size = 128; % the batch size
elseif netID == 2
    network = 'caffe_reference_places205';
    layers = {'conv5','conv4','conv3','conv2','conv1'};
    batch_size = 128;
elseif netID == 3
    network = 'caffe_reference_imagenetplaces205';
    layers = {'conv5','conv4','conv3','conv2','conv1'};
    batch_size = 128;
elseif netID == 4
    network = 'vgg16_places365'
    layers = {'conv5_3','conv5_2','conv5_1','conv4_3','conv4_2','conv4_1','conv3_3','conv3_2','conv3_1','conv2_2','conv2_1','conv1_1'};
    batch_size = 32;
end


net_prototxt = sprintf('%s/%s.prototxt', root_model, network);
net_binary = sprintf('%s/%s.caffemodel', root_model, network);

%% standard setup caffe
use_gpu = 1;

if(use_gpu)
    caffe.set_mode_gpu();
    caffe.set_device(device_id);
else
    caffe.set_mode_cpu();
end

net = caffe.Net(net_prototxt, net_binary, 'test');

% Load images in parallel
if matlabpool('size')==0
    try
        matlabpool(6)
    catch e
    end
end

% Get the network architecture information
layernames = net.blob_names;
netInfo = cell(size(layernames,1),3);
for i=1:size(layernames,1)
    netInfo{i,1} = layernames{i};
    netInfo{i,2} = i;
    tmp = net.blobs(layernames{i}).shape;
    if tmp(1) == 1
        tmp = tmp(3:end);
    end
    netInfo{i,3} = tmp;
end

IMAGE_MEAN = caffe.io.read_mean('model/places_mean.binaryproto');
CROPPED_DIM = netInfo{1,3}(1); % alexNet is 227, googlenet input is 224
IMAGE_MEAN = imresize(IMAGE_MEAN,[CROPPED_DIM CROPPED_DIM]);
    
num_batches = ceil(num_images / batch_size);
    
%% feature extraction step
num_layers = numel(layers);
num_units_layers = zeros(num_layers,1);
features_CNN = cell(num_layers,1); % the features
weights_CNN = cell(num_layers,1); % the parameters(weight) of each unit
 
for i=1:num_layers
    layerID = find(strcmp(netInfo(:,1),layers{i}) == 1);
    activation_struct = netInfo{layerID,3};
    param_layer = net.params(layers{i},1).get_data();
    activation_layer = net.blobs(layers{i}).get_data();
    weights_CNN{i} = param_layer;
    if size(activation_layer, 3) == 1 
        num_units = size(activation_layer, 1);
        feature_layer = zeros(num_images, num_units,  'single'); % FC layer
    else 
        num_units = size(activation_layer, 3);
        feature_layer = zeros(num_images, size(activation_layer,3), size(activation_layer,1), size(activation_layer,2), 'single'); % spatial conv layer [num_images, num_unit, H, W], this variable could be very large, which results to Out Of Memory error in matlab.
    end
    features_CNN{i} = feature_layer;
    num_units_layers(i) = num_units;
end

% reset the batch_size
inputSize_default = net.blobs('data').shape;
net.blobs('data').reshape([inputSize_default(1) inputSize_default(2) inputSize_default(3) batch_size]);


for curBatchID=1:num_batches
    [imBatch] = generateBatch( imageList(:,1), curBatchID, batch_size, num_batches, IMAGE_MEAN, CROPPED_DIM);
    scores = net.forward({imBatch});
    curStartIDX = (curBatchID-1)*batch_size+1;
    if curBatchID == num_batches
        curEndIDX = num_images;
    else
        curEndIDX = curBatchID*batch_size;
    end
    for layerID = 1:num_layers
        features_batch = net.blobs(layers{layerID}).get_data();
        if size(features_batch,4) == 1
            features_batch = features_batch';
            features_CNN{layerID}(curStartIDX:curEndIDX,:) = features_batch(1:curEndIDX - curStartIDX + 1);
        else
            features_batch = permute(features_batch, [4 3 2 1]); % reshuffle this to [batch_size, num_units, H, W]
            features_CNN{layerID}(curStartIDX:curEndIDX,:,:,:) = features_batch(1:curEndIDX - curStartIDX + 1, :, :, :);
        end
    end
    
    disp([network  ' feature extraction:' num2str(curBatchID) '/' num2str(num_batches)]);
end   
file_save = fullfile(root_features,sprintf('features_%s_%s.mat', data_name, network));
disp(sprintf('features are output to %s', file_save));
save(file_save,'features_CNN','layers','netInfo', 'imageList', '-v7.3')
    
caffe.reset_all()     
