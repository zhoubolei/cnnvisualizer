function bases = buildGradientBases(img, method, param)
%
% B = buildGradientBases(img, method, param)
%
%  methods:
%     grid - rectangular windows
%        param.L = size of the window
%
%     superpixels
%
%     edges
%        param.minlength = 20;
%
%
% The input image can be recovered as:
%   figure; imshow(sum(B,4), [])

if isfield(param,'visualize_bases') == 1
    visualize_bases = param.visualize_bases;
else
    visualize_bases = 0;    
end

if nargin < 2
    param.L = 4;
    method = 'grid';
end

[nrows, ncols, channels] = size(img);
fn = getFilters; % dx, dy
Nfilters = size(fn,3);

% get basis masks
switch method
    % method 1: grid
    case 'grid'
        M = zeros([nrows ncols 1000]);
        b = 0;
        for x = 0:param.L:ncols-param.L
            for y = 0:param.L:nrows-param.L
                b = b+1;
                M(y+1:y+param.L, x+1:x+param.L, b) = 1;
            end
        end
        M = M(:,:,1:b);
        
    case 'superpixels'
        % method 2: segments
        M = getSuperPixels(img, param);
        
    case 'edges'
        % method 3: edges
        %   Gradient domain editing+integration allows keeping large regions with different contrast between inside and outside 
        % while removing the texture inside the object. This is not
        % possible taking patches. 
        M = getEdgeMasks(img, param);
        
    case 'corners'
        % method 3: edges
        %   Gradient domain editing+integration allows keeping large regions with different contrast between inside and outside 
        % while removing the texture inside the object. This is not
        % possible taking patches. 
        M = getCornerMasks(img, param);
        fprintf('There are %d corner basis. \n', size(M,3))
        
    case 'edgesandregions'
        Me = getEdgeMasks(img, param);
        Ms = getSuperPixels(img, param);
        
        m = sum(Me,3);
        Ms = Ms .*repmat((1-m), [1 1 size(Ms,3)]);

        M = cat(3, Me, Ms);
        fprintf('There are %d edge basis. \n', size(Me,3))
        fprintf('There are %d region basis. \n', size(Ms,3))
        
    case 'edgesandregionsandcorners'
        Mc = getCornerMasks(img, param);
        Me = getEdgeMasks(img, param);
        Ms = getSuperPixels(img, param);
        
        % remove corners from edges:
        mc = sum(Mc,3);
        Me = Me .*repmat((1-mc), [1 1 size(Me,3)]);

        % remove corners and edges from regions
        me = sum(Me,3);
        Ms = Ms .*repmat((1-me).*(1-mc), [1 1 size(Ms,3)]);

        M = cat(3, Mc, cat(3, Me, Ms));
        
        disp(sprintf('There are %d Bases: %d corners, %d edges and %d regions', size(M,3), size(Mc,3), size(Me,3), size(Ms,3)))
end



Nbases = size(M,3);

% obtain gradients
for c = 1:channels
    out(:,:,:,c) = convFn(img(:,:,c), fn);
end

% Loop over masks and get basis bunctions
B = zeros([nrows ncols channels Nbases]);
suport=zeros(Nbases,1);
for b = 1:Nbases
    suport(b) = sum(sum(M(:,:,b)));
    % Integration
    m = repmat(M(:,:,b), [1 1 Nfilters]);
    for c = 1:size(img,3)
        g = out(:,:,:,c).*m;
        B(:,:,c,b)=deconvFn(g, fn);
       % B(:,:,c,b) = B(:,:,c,b)-mean(mean(B(:,:,c,b)));
    end
end

bases.B = B;
bases.Masks = M;
bases.Nbases = Nbases;
bases.method = method;
bases.param = param;
bases.channel_means = [mean(mean(img(:,:,1))) mean(mean(img(:,:,2))) mean(mean(img(:,:,3)))];

if visualize_bases
    showBases(bases)
end



function M = getSuperPixels(img, param)
[nrows, ncols, channels] = size(img);

ratio = 2; %  tradeoff between color importance and spatial importance (larger values give more importance to color),
kernelsize = 2; % kernelsize is the size of the kernel used to estimate the density
maxdist = param.maxdist; % maxdist is the maximum distance between points in the feature space that may be linked if the density is increased.
[~, labels] = vl_quickseg(uint8(img), ratio, kernelsize, maxdist);

Nbases = max(labels(:));
M = zeros([nrows ncols Nbases]);
for b = 1:Nbases
    M(:,:,b) = (labels==b);
end

        
function M = getEdgeMasks(img, param)
[nrows, ncols, channels] = size(img);


edgeim = edge(mean(img,3),'canny', [0.1 0.2], 1);
[~, edgeim] = edgelink(edgeim, param.minlength);
list = unique(edgeim(:));
list = list(2:end);

Nbases = length(list);
M = zeros([nrows ncols Nbases]);
b=0;
L = 7;
w = hamming(L)*hamming(L)';
w = double(w>=w(1,(L+1)/2));
for bb = 1:Nbases
    tmp = double(edgeim==list(bb));
    if sum(tmp(:))>param.minlength
        b = b+1;
        tmp = conv2(tmp, w, 'same')>0;
        M(:,:,b) = tmp;
    end
end
Nbases = b;
M = double(M(:,:,1:Nbases));

% loop to get the max ()
M = M./repmat(eps+sum(M,3), [1 1 Nbases]);


function M = getCornerMasks(img, param)
[nrows, ncols, channels] = size(img);

C = corner(mean(img,3), 'MinimumEigenvalue', param.Ncorners);
Nbases = size(C,1);

M = zeros([nrows, ncols Nbases]);
for c = 1:size(C,1)
    M(C(c,2),C(c,1),c) = 1;
end

L = 7;
w = hamming(L)*hamming(L)';
w = double(w>=w(1,(L+1)/2));

M = convn(M, w, 'same');

% loop to get the max ()
M = M./repmat(eps+sum(M,3), [1 1 Nbases]);

