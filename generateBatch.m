function [imBatch] = generateBatch( images, curBatchID, batch_size, num_batches, image_mean, IMAGE_DIM)
%GENERATEBATCH Summary of this function goes here
%   Detailed explanation goes here
curStartIDX = (curBatchID-1)*batch_size+1;
if curBatchID == num_batches
    curEndIDX = size(images,1);
else
    curEndIDX = curBatchID*batch_size;
end

imBatch = zeros(IMAGE_DIM, IMAGE_DIM, 3, batch_size, 'single');

nIter = curEndIDX-curStartIDX+1;

parfor i=1:nIter
    try 
        im = imread(images{i+curStartIDX-1,1});
    catch exception
        disp(['error reading ' images{i+curStartIDX-1,1}])
        im = zeros(256,256);
    end
    if size(im,3)==1
        im = repmat(im,[1 1 3]);
    end     
    im = single(im);
    im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
    im = im(:,:,[3 2 1]) - image_mean;
    imBatch(:,:,:,i) = permute(im, [2 1 3]);
end


end
