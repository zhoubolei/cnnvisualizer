% BILINIMIX  An Interactive Image for viewing multiple images
%
% Usage:  bilinimix(im, figNo)
%
% Arguments:  im - 2D Cell array of images to be blended.  I
%          figNo - Optional figure window number to use.
%
% This function provides an 'Interactive Image'.  It is intended to allow
% efficient visual exploration of a sequence of images that have been processed
% with a series of two different parameter values, for example, scale and image
% mix.  The horizontal and vertical positions of the cursor within the image
% controlling the bilinear blend of the two parameters over the images.  
%
% Click in the image to toggle in/out of blending mode.  Move the cursor 
% within the image to blend between the input images.
%
% See also: LINIMIX, LOGISTICWEIGHTING
%
% Note this code is rather rough and needs a good cleanup but is still
% reasonably useful at this stage

% Copyright (c) 2012 Peter Kovesi
% Centre for Exploration Targeting
% The University of Western Australia
% peter.kovesi at uwa edu au
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in 
% all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.
%
% April 2012

function  bilinimix(im, figNo)
    
    assert(iscell(im), 'Input images must be in a cell array');
    [gridRows gridCols] = size(im);
    
    nImages = gridRows * gridCols;
    
    fprintf('\nClick in the image to toggle in/out of blending mode \n');
    fprintf('Move the cursor up and down within the image to \n');
    fprintf('blend between the input images\n\n');
        
    % Check sizes of images
    % Ensure every image is stored as rgb, even if it
    % is greyscale. ** To Do: But if they are all greyscale do not do this **
    rows = zeros(nImages,1); cols = zeros(nImages,1); chan = zeros(nImages,1);
    for n = 1:nImages
        [rows(n) cols(n), chan(n)] = size(im{n});
        if 0 % chan(n) == 2
            im{n} = repmat(im{n},[1 1 3]);  % *** Replicate image in R, G and B channels
        end
    end

    % Trim all images down to the size of the smallest image and normalise to
    % range 0-1.
    minrows = min(rows); mincols = min(cols);
    for n = 1:nImages
      im{n} = normalise(im{n}(1:minrows, 1:mincols, :));    
    end
    
    % Get handle to Cdata in the figure
    if exist('figNo','var')
        fig = figure(figNo);
    else
        fig = figure;
    end
    
    S = warning('off');
    imshow(im{ max(1,fix(nImages/2)) }, 'border', 'tight');
    drawnow
    warning(S)
    
%    set(fig, 'NumberTitle', 'off')
    set(fig, 'name', 'CET Image Blender')
    ah = get(fig,'CurrentAxes');
    imHandle = get(ah,'Children'); 
    
    set(fig,'WindowButtonDownFcn',@wbdcb);
    set(fig,'Menubar','none');
    blending = 0;
    
    % Set up custom pointer
    myPointer = [circularstruct(7) zeros(15,1); zeros(1, 16)];
    myPointer(~myPointer) = NaN;

    hspot = 0;
    hold on
    
%-----------------------------------------------------------------------    
% Window button down callback
function wbdcb(src,evnt)
    if strcmp(get(src,'SelectionType'),'normal')
        if ~blending  % Turn blending on
            blending = 1;
%            set(src,'Pointer','cross')
            set(src,'Pointer','custom', 'PointerShapeCData', myPointer,...
                    'PointerShapeHotSpot',[9 9])
            set(src,'WindowButtonMotionFcn',@wbmcb)
            
            if hspot  % For paper illustration
                delete(hspot);
            end
            
        else         % Turn blending off
            blending = 0;
            set(src,'Pointer','arrow')
            set(src,'WindowButtonMotionFcn','')
            
            % For paper illustration
            cp = get(ah,'CurrentPoint');
            x = cp(1,1);
            y = cp(1,2);
            hspot = plot(x, y, '.', 'color', [0 0 0], 'Markersize', 40');
        end
    end
end 

%-----------------------------------------------------------------------    
% Window button move call back
function wbmcb(src,evnt)
    cp = get(ah,'CurrentPoint');
    x = cp(1,1); y = cp(1,2);
    blend(x, y)
end

%-----------------------------------------------------------------------
function blend(x, y)
    
    % Clamp x and y to image limits
    x = max(0,x); x = min(mincols,x);  
    y = max(0,y); y = min(minrows,y);  
    
    % Compute grid coodinates of (x,y)
    % im{1,1}        im{1,2} ... im{1, gridCols}
    %    ..             ..              ..
    % im{gridRows,1}    ..   ... im{gridRows, gridCols}
    gx = x/mincols * (gridCols-1) + 1;
    gy = y/minrows * (gridRows-1) + 1;
    
    % Compute bilinear interpolation between the images
    gxf = floor(gx); gxc = ceil(gx);
    gyf = floor(gy); gyc = ceil(gy);

    frac = gxc-gx;
    blendyf = (frac)*im{gyf,gxf} + (1-frac)*im{gyf,gxc};
    blendyc = (frac)*im{gyc,gxf} + (1-frac)*im{gyc,gxc};

    frac = gyc-gy;
    blendim = (frac)*blendyf + (1-frac)*blendyc;
    
    set(imHandle,'CData', blendim);
    
end 

%---------------------------------------------------------------------------
end % of bilinimix
    
