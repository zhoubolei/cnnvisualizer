function [layers, batch_size ] = return_network(network)

if network == 'caffe_reference_imagenet'
    layers = {'fc8','fc7','fc6','conv5','conv4','conv3','conv2','conv1'};
    batch_size = 128; % the batch size
elseif network == 'caffe_reference_places205'
    layers = {'fc8','fc7','fc6','conv5','conv4','conv3','conv2','conv1'};
    batch_size = 128;
elseif network == 'caffe_reference_imagenetplaces205'
    layers = {'fc8','fc7','fc6','conv5','conv4','conv3','conv2','conv1'};
    batch_size = 128;
elseif network == 'caffe_reference_places365'
    layers = {'fc8','fc7','fc6','conv5','conv4','conv3','conv2','conv1'};
    batch_size = 128;
elseif network == 'vgg16_imagenet'
    layers = {'fc8', 'fc7','fc6','conv5_3','conv5_2','conv5_1','conv4_3','conv4_2','conv4_1','conv3_3','conv3_2','conv3_1','conv2_2','conv2_1','conv1_1'};
    batch_size = 32;
elseif network == 'vgg16_places205'
    layers = {'fc8', 'fc7', 'fc6','conv5_3','conv5_2','conv5_1','conv4_3','conv4_2','conv4_1','conv3_3','conv3_2','conv3_1','conv2_2','conv2_1','conv1_1'};
    batch_size = 32;
elseif network = 'vgg16_places365'
    layers = {'fc8a', 'fc7', 'fc6','conv5_3','conv5_2','conv5_1','conv4_3','conv4_2','conv4_1','conv3_3','conv3_2','conv3_1','conv2_2','conv2_1','conv1_1'};
    batch_size = 32;
elseif network = 'vgg16_hybrid1365'
    layers = {'fc8a', 'fc7', 'fc6','conv5_3','conv5_2','conv5_1','conv4_3','conv4_2','conv4_1','conv3_3','conv3_2','conv3_1','conv2_2','conv2_1','conv1_1'};
    batch_size = 32;
else
    disp('no such networks')
    batch_size = 0
    layers = {}
end


