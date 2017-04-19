% ---------------------------------------------------
% Sample code to use synthetic receptive field of unit to segment the image.
%
% here we first cache 5 top responded images and their conv5 feature maps for the unit 49 at conv5 of Places-CNN, then we use the sythetic receptive field of the unit to segment the image based on the feature map for each image.
%
% Bolei Zhou
% 2015/5/22
% ---------------------------------------------------


%% generate uniform receptive field
RFsize = 65;                    % the average actual size of conv5, you could change it to get a tighter segmentation
para.gridScale = [13 13];       % conv5 of alexNet feature map
para.imageScale = [227 227];    % the input image size
para.RFsize = [RFsize RFsize];  
para.plotPointer = 0;           % whether to show the generated RF
maskRF = generateRF( para);



thresholdSegmentation = 0.5;    % segmentation threshold

unitID = 49;                    % conv5 unit49

figure
for i=1:5
    curImg = imread(['data/unitID' num2str(unitID) '_img' num2str(i) '.jpg']);
    curImg = im2double(imresize(curImg,para.imageScale));
    curFeatureMap = load(['data/unitID' num2str(unitID) '_feature' num2str(i) '.mat']); % the extracted feature map for unit 49 at conv5 of places-CNN.
    curFeature_vectorized = curFeatureMap.featureMap(:);
    maxValue = max(curFeature_vectorized);
    IDX_max = find(curFeature_vectorized>maxValue * thresholdSegmentation);
    curMask = squeeze(sum(maskRF(IDX_max,:,:),1));
    curMask(curMask>0) = 1;

    IDX_region = find(curMask>0);
    curSegmentation = repmat(curMask,[1 1 3]).*curImg+0.2*(1- repmat(curMask,[1 1 3])).*curImg;

    
    subplot(1,5,i),imshow(curSegmentation);
end
