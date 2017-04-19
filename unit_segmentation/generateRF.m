function [maskRF] = generateRF( para)
% function to generate synthetic Receptive field
%   
maskRF = maskRound(para);
plotPointer = para.plotPointer;
if plotPointer == 1
    for i=1:size(maskRF,1);
        curMask = squeeze(maskRF(i,:,:));
        imshow(curMask);title(['RF IDX:' num2str(i)])
        waitforbuttonpress
    end
end

end

function maskRF = maskRectangle(para)
gridScale = para.gridScale;
imageScale = para.imageScale;
RFsize = para.RFsize;

maskRF = zeros(gridScale(1)*gridScale(2), imageScale(1),imageScale(2));
RFsizeX = RFsize(1);
RFsizeY = RFsize(2);
interval = floor(imageScale(1)./(gridScale(1)-1));
halfRFsizeX = round(RFsizeX/2);
halfRFsizeY = round(RFsizeY/2);
for i=1:gridScale(1)
    for j=1:gridScale(2)
        curGridAxisY = (i-1)*interval+1;
        curGridAxisX = (j-1)*interval+1;
        curMask = zeros(imageScale);
        curXrange = [max(1, curGridAxisX-halfRFsizeX),min(imageScale(1), curGridAxisX+halfRFsizeX)];
        curYrange = [max(1, curGridAxisY-halfRFsizeY),min(imageScale(2), curGridAxisY+halfRFsizeY)];
        curMask(curXrange(1):curXrange(2),curYrange(1):curYrange(2)) = 1;
        curIDX = (i-1)*gridScale(1) + j;
        maskRF(curIDX,:,:) = curMask;
    end
end

end

function maskRF = maskRound(para)
convFilter = fspecial('disk', round(para.RFsize(1)/2));


gridScale = para.gridScale;
imageScale = para.imageScale;
RFsize = para.RFsize;

maskRF = zeros(gridScale(1)*gridScale(2), imageScale(1),imageScale(2),'single');
RFsizeX = RFsize(1);
RFsizeY = RFsize(2);
interval = floor(imageScale(1)./(gridScale(1)-1));

for i=1:gridScale(1)
    for j=1:gridScale(2)
        curGridAxisY = (i-1)*interval+1;
        curGridAxisX = (j-1)*interval+1;
        curMask = zeros(imageScale,'single');
        curMask(curGridAxisX,curGridAxisY) = 100;
        curMask = imfilter(curMask, convFilter);
        curMask(curMask>0.01) = 1;
        curIDX = (i-1)*gridScale(1) + j;
        maskRF(curIDX,:,:) = curMask;
    end
end

end