
% ------------------------------------------------------------------------
% Coded by Agata Lapedriza and Antonio Torralba
% MIT, 2015
% ------------------------------------------------------------------------

addpath(genpath(pwd));

% I = double(imread('coast_nat120.jpg'));
I = double(imread('kitchen.jpg'));
I = imresize(I,256/min(size(I,1), size(I,2)));

figure
imshow(uint8(I));

% select the method to create the bases
% method = 'edgesandregionsandcorners';
% method = 'edges';
 method = 'edgesandregions';
%method =  'superpixels';

% bases parameters
param.minlength = 70; % example: 2; for edges (larger=> less edges)
param.maxdist = 60;%50; % example: 10 in grey scale, 60 in rgb; for segments (larger => less segments)
param.Ncorners = 10; % example: 10; number of corners



% generate bases --------------------------------------------------------
bases = buildGradientBases(I, method, param);
showBases(bases);

% generate a random reconstruction --------------------------------------
w = rand(1,bases.Nbases) < 0.3;
Ihat = reconstruct(bases,w);
Ihat_complement = reconstruct(bases,w == 0);
figure
subplot(1,2,1)
imshow(uint8(Ihat));
subplot(1,2,2)
imshow(uint8(Ihat_complement));

% generate N random reconstructions --------------------------------------
Nimages = 20;
figure
for i = 1:Nimages
    w = rand(1,bases.Nbases) < 0.3;
    %w = ones(size(w));
    Ihat = reconstruct(bases, w);
    subplot(4,5,i)
    imshow(uint8(Ihat));
    
    % imwrite(uint8(Ihat), sprintf('tests/test_%d.jpg',i))
end

% select bases to get a visualitzation ----------------------------------
% select all the bases you want and then click to the red button
guiSelectBasis;
Ihat = reconstruct(bases,selected);
Ihat_complement = reconstruct(bases,selected == 0);
figure
subplot(1,2,1)
imshow(uint8(Ihat));
subplot(1,2,2)
imshow(uint8(Ihat_complement));
imwrite(uint8(Ihat), 'tests/bases_selected.jpg');
imwrite(uint8(Ihat_complement), 'tests/bases_selected_compl.jpg');