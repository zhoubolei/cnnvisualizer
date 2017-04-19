%% sample code to generate the top ranked image segmentation using the activation maps for each unit.

addpath('/data/vision/torralba/gigaSUN/caffe-cuda8.0/matlab');

%% the candidate image set: in the ICLR'15 experiments, we combine the images of SUN397 and the images of ILSVRC'12 validation set. You could use your own image set.

target_folder = 'result_segments';
imageList = textread('dataset/imglist_unisegV4.txt','%s');
root_dataset = 'dataset/uniseg4_384';
nImgs = numel(imageList);
for i=1:nImgs
    imageList{i} = fullfile(root_dataset, imageList{i});
end


if ~exist(target_folder)
    mkdir(target_folder)
end


net_batch = 1;
device_id = 1;
        
if net_batch == 1
    net_names = {
        'imagenet_alexnet'
        'places205_alexnet'
    };
    net_prototxts = {
        'model/alexnet_deploy.prototxt'
        'model/alexnet_deploy.prototxt'
    }; 
    net_binaries = {
        'model/caffe_reference_imagenet.caffemodel'
        'model/caffe_reference_places205.caffemodel'
    };    
    
elseif net_batch == 2
    net_names = {
        'places365_alexnet'
    };
    net_prototxts = {
        'model/alexnet_deploy.prototxt'
    };
    net_binaries = {
        'model/caffe_reference_places365.caffemodel'
    };
elseif net_batch == 3
    net_names = {
        'places205_vgg16'
    };
    net_prototxts = {
        'model/vgg16_deploy.prototxt'
    };
    net_binaries = {
        'model/vgg16_places205.caffemodel'
    };
end

%% standard setup caffe
use_gpu = 1;

if(use_gpu)
    caffe.set_mode_gpu();
    caffe.set_device(device_id);
else
    caffe.set_mode_cpu();
end

%% check the model and prototxt could beloaded
for i = 1:numel(net_names)
    name_current = net_names{i};
    prototxt_current = fullfile(net_prototxts{i});
    binary_current = net_binaries{i};
    net = caffe.Net(prototxt_current, binary_current, 'test');
    disp([name_current ' verified']);
    caffe.reset_all() 
end

%% the feature extraction and image segmentation process
cropSize = [150 150];
topNum = 30; 
inputImg_size = [224 224];
threshold_segment = 0.4; % it controls the segmentation

% loading images in parallel
if matlabpool('size')==0
    try
        matlabpool(6)
    catch e
    end
end

for modelID = 1:numel(net_names)
    name_current = net_names{modelID};
    prototxt_current = net_prototxts{modelID};
    binary_current = net_binaries{modelID};
    net = caffe.Net(prototxt_current, binary_current, 'test');

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
    
    num_units = netInfo{end,3}(3);
    %% feature extraction step
    feature_unitMax = zeros(numel(imageList), num_units, 'single'); % record the max value
    for curBatchID=1:num_batches
        [imBatch] = generateBatch( imageList(:,1), curBatchID, batch_size, num_batches, IMAGE_MEAN, CROPPED_DIM);
        scores = net.forward({imBatch});
        curStartIDX = (curBatchID-1)*batch_size+1;
        if curBatchID == num_batches
            curEndIDX = nImgs;
        else
            curEndIDX = curBatchID*batch_size;
        end
        curFeatures_batch = scores{1};
        curFeatures_batch = max(curFeatures_batch,[],1);
        curFeatures_batch = max(curFeatures_batch,[],2);
        curFeatures_batch = squeeze(curFeatures_batch);
        feature_unitMax(curStartIDX:curEndIDX,:) = curFeatures_batch(:, 1:curEndIDX-curStartIDX+1)';   
        
        max_batch = max(curFeatures_batch,[],2);
        disp([name_current  ' feature extraction:' num2str(curBatchID) '/' num2str(num_batches)]);
    end 
    
    
    %% start segmentation 

    saveFolder = fullfile(target_folder, name_current);
    
    if ~exist(saveFolder)
        mkdir(saveFolder);
    end
    fileName = [saveFolder '.html'];
    fp = fopen(fileName,'w');
    fprintf(fp,'<html>\n');
    fprintf(fp,'<head><style> img { height: 150px;} </style></head>\n');
    fprintf(fp,'<body>\n');
    fprintf(fp,'<hr/>');
    fprintf(fp,'<h2>%s</h2>\n', name_current);
    fprintf(fp,'<p>%s<br>%s</p>\n',prototxt_current, binary_current);
    for unitID = 1:num_units
        fprintf(fp,'<br>%s<br>\n',['unit ' num2str(unitID)]);
        fprintf(fp,'<img src="%s" />\n', fullfile(name_current, ['unitID' num2str(unitID) '.jpg']));
    end
    fprintf(fp,'<hr/>');
    fprintf(fp,'</body></html>\n');
    fclose(fp);
    disp([fileName])
    %% unit segmentation step
    
    for unitID = 1:num_units
        curFeatureMax = feature_unitMax(:, unitID);
        [maxValue_sorted, imgIDX_sorted] = sort(curFeatureMax, 'descend');
        curSegmentation = [];
        
        % select on the top ranked images
        imageList_top = imageList(imgIDX_sorted(1:batch_size));
        
        
        [imBatch] = generateBatch( imageList_top, 1, batch_size, num_batches, IMAGE_MEAN, CROPPED_DIM);
        scores = net.forward({imBatch});
        scores = scores{1};
        
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
            curGridResponse  = squeeze(scores(:, :, unitID, imgID))';
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
        imwrite(curSegmentation, [saveFolder '/unitID' num2str(unitID) '.jpg']);
        disp([name_current ' segmenting unitID' num2str(unitID)]);
    end
    caffe.reset_all()     
end

