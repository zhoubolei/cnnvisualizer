%% sample code to generate the top ranked image segmentation using the activation maps for each unit for all the layers

addpath('yourpath/caffe/matlab');

% load the sample dataset
imageList = textread('images/imagelist.txt','%s');
root_dataset = 'images';
nImgs = numel(imageList);
for i=1:nImgs
    imageList{i} = fullfile(root_dataset, imageList{i});
end



device_id = 0;
       
zoo_path = 'models';
netID = 2;
if netID == 1
    network = 'caffe_reference_places365';
    layers = {'conv5','conv4','conv3','conv2','conv1'};
elseif netID == 2
    network = 'vgg16_places365'
    layers = {'conv5_3','conv5_2','conv5_1','conv4_3','conv4_2','conv4_1','conv3_3','conv3_2','conv3_1','conv2_2','conv2_1','conv1_1'};
end



net_prototxt = sprintf('%s/%s.prototxt', zoo_path, network);
net_binary = sprintf('%s/%s.caffemodel', zoo_path, network);

%% standard setup caffe
use_gpu = 1;

target_folder = fullfile('result_segments', network);
if ~exist(target_folder)
    mkdir(target_folder)
end

if(use_gpu)
    caffe.set_mode_gpu();
    caffe.set_device(device_id);
else
    caffe.set_mode_cpu();
end

net = caffe.Net(net_prototxt, net_binary, 'test');

%% the feature extraction and image segmentation process
cropSize = [150 150];
topNum = 30; 
inputImg_size = [224 224];
threshold_segment = 0.4; % it controls the tightness of the segmentation

% loading images in parallel
if matlabpool('size')==0
    try
        matlabpool(6)
    catch e
    end
end

if ~exist('layers_unitMax','var')
    % get the network architecture
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

    batch_size = netInfo{1,3}(4);
    num_batches = ceil(nImgs / batch_size);
    
    %% feature extraction step
    num_layers = numel(layers);
    layers_unitMax = cell(num_layers,1);
    num_units_layers = zeros(num_layers,1);
    for i=1:num_layers
        layerID = find(strcmp(netInfo(:,1),layers{i}) == 1);
        activation_struct = netInfo{layerID,3};
        param_layers = net.params(layers{i},1).get_data();
        num_unit = size(param_layers, 4);
        feature_unitMax = zeros(numel(imageList), num_unit, 'single'); % record the max value
        layers_unitMax{i} = feature_unitMax;
        num_units_layers(i) = num_unit;
    end

    for curBatchID=1:num_batches
        [imBatch] = generateBatch( imageList(:,1), curBatchID, batch_size, num_batches, IMAGE_MEAN, CROPPED_DIM);
        scores = net.forward({imBatch});
        curStartIDX = (curBatchID-1)*batch_size+1;
        if curBatchID == num_batches
            curEndIDX = nImgs;
        else
            curEndIDX = curBatchID*batch_size;
        end
        for layerID = 1:num_layers
            curFeatures_batch = net.blobs(layers{layerID}).get_data();
            curFeatures_batch = max(curFeatures_batch,[],1);
            curFeatures_batch = max(curFeatures_batch,[],2);
            curFeatures_batch = squeeze(curFeatures_batch);
            layers_unitMax{layerID}(curStartIDX:curEndIDX,:) = curFeatures_batch(:, 1:curEndIDX-curStartIDX+1)';   
        end
        disp([network  ' feature extraction:' num2str(curBatchID) '/' num2str(num_batches)]);
    end 
    save(sprintf('unitMax_%s.mat', network),'layers_unitMax','layers', '-v7.3')
end
    
%% start segmentation given the top activations of each image.
for layerID = 1:numel(layers)
    name_current = layers{layerID};
    saveFolder = fullfile(target_folder, 'image');
    
    if ~exist(saveFolder)
        mkdir(saveFolder);
    end
    fileName = sprintf('%s/%s.html', target_folder, layers{layerID});

    fp = fopen(fileName,'w');
    fprintf(fp,'<html>\n');
    fprintf(fp,'<head><style> img { height: 150px;} </style></head>\n');
    fprintf(fp,'<body>\n');
    fprintf(fp,'<hr/>');
    fprintf(fp,'<h2>%s</h2>\n', name_current);
    for unitID = 1:num_units_layers(layerID)
        fprintf(fp,'<br>%s<br>\n',['unit ' num2str(unitID)]);
        fprintf(fp,'<img src="%s" />\n', fullfile('image', sprintf('%s-%04d.jpg', layers{layerID}, unitID-1)));% 0 index
    end
    fprintf(fp,'<hr/>');
    fprintf(fp,'</body></html>\n');
    fclose(fp);
    disp(fileName)
    %% unit segmentation step
    for unitID = 1:num_units_layers(layerID)
        curFeatureMax = layers_unitMax{layerID}(:, unitID);
        [maxValue_sorted, imgIDX_sorted] = sort(curFeatureMax, 'descend');
        curSegmentation = [];
        
        % select on the top ranked images
        imageList_top = imageList(imgIDX_sorted(1:batch_size));
        
        
        [imBatch] = generateBatch( imageList_top, 1, batch_size, num_batches, IMAGE_MEAN, CROPPED_DIM);
        scores = net.forward({imBatch});
        scores = scores{1};
        curFeatures_batch = net.blobs(layers{layerID}).get_data();
        for imgID = 1:min(topNum, batch_size)
            
            try 
                curImg = imread(imageList_top{imgID}); 
            catch exception
                curImg = ones(256,256,3);
            end
            curImgShow = imresize(im2double(curImg),inputImg_size);
            if size(curImgShow,3) == 1
                curImgShow = repmat(curImgShow,[1 1 3]);
            end
            curGridResponse  = squeeze(curFeatures_batch(:, :, unitID, imgID))';
            curGridResponse = abs(curGridResponse);
            
            curGridResponse = imfilter(curGridResponse, fspecial('average'));
            curGridResponse = curGridResponse./max(curGridResponse(:));
            
            curMask = imresize(curGridResponse, inputImg_size);
            
            curMask(curMask>threshold_segment) = 1; % 0.2 for other network, 0.5 for googlenet
            curMask(curMask<threshold_segment) = 0;
            curImgResult = repmat(curMask,[1 1 3]).*curImgShow+0.2*(1- repmat(curMask,[1 1 3])).*curImgShow;
            curImgResult = imresize(curImgResult,cropSize);
            curSegmentation = [curSegmentation ones(size(curImgResult,1),3,3) curImgResult];
        end
        imwrite(curSegmentation, sprintf('%s/%s-%04d.jpg', saveFolder, layers{layerID}, unitID-1));
        disp([layers{layerID} ' segmenting unitID' num2str(unitID)]);
    end
end
caffe.reset_all()     
