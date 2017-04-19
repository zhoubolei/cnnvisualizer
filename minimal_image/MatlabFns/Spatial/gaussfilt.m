% GAUSSFILT -  Small wrapper function for convenient Gaussian filtering
%
% Usage:  smim = gaussfilt(im, sigma)
%
% Arguments:   im - Image to be smoothed.
%           sigma - Standard deviation of Gaussian filter.
%
% Returns:   smim - Smoothed image.
%
% See also:  INTEGGAUSSFILT

% Peter Kovesi
% Centre for Explortion Targeting
% The University of Western Australia
% http://www.csse.uwa.edu.au/~pk/research/matlabfns/

% March 2010
% June  2013  - Provision for multi-channel images

function smim = gaussfilt(im, sigma)
 
    if sigma == 0  
        smim = im;
        return;
    end
    
    % If needed convert im to double
    if ~strcmp(class(im),'double')
        im = double(im);  
    end
    
    sze = ceil(6*sigma);  
    if ~mod(sze,2)    % Ensure filter size is odd
        sze = sze+1;
    end
    sze = max(sze,1); % and make sure it is at least 1
    
    h = fspecial('gaussian', [sze sze], sigma);

    % Apply filter to all image channels
    smim = zeros(size(im));
    for n = 1:size(im,3)
        smim(:,:,n) = filter2(h, im(:,:,n));
    end
    