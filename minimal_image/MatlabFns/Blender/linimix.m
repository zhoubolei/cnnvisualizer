% LINIMIX  An Interactive Image for viewing multiple images
%
% Usage:  linimix(im, B, figNo)
%
% Arguments:  im - 1D Cell array of images to be blended.  If this is not
%                  supplied, or is empty, the user is prompted with a file
%                  dialog to select a series of images.
%              B - Parameter controlling the weighting function when
%                  blending between two images.
%                  B = 0    Linear transition between images (default)
%                  B = 2^4  Sigmoidal transition at midpoint.
%                  B = 2^10 Near step-like transition at midpoint.
%          figNo - Optional figure window number to use.
%
% This function provides an 'Interactive Image'.  It is intended to allow
% efficient visual exploration of a sequence of images that have been processed
% with a series of different parameter values, for example, scale.  The vertical
% position of the cursor within the image controls the linear blend between
% images.  With the cursor at the top the first image is displayed, at the
% bottom the last image is displayed. At positions in between blends of
% intermediate images are dislayed.  
%
% Click in the image to toggle in/out of blending mode.  Move the cursor up and
% down within the image to blend between the input images.
%
% Use BILINIMIX if you want the horizontal position of the cursor to be used
% too.  This will allow visual exploration of a sequence of images controlled by
% two different processing parameters.  Alternatively one could blend between
% images of two different modalities over some varing parameter, say scale.
%
% See also: BILINIMIX, LOGISTICWEIGHTING

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
% March 2012

function  linimix(im, B, figNo, XY)

    if ~exist('im', 'var'), im = []; end
    [im, nImages, fname] = collectncheckimages(im);

    if ~exist('B', 'var'), B = 0; end
    if ~exist('XY', 'var'), XY = 'Y'; end
    
    fprintf('\nClick in the image to toggle in/out of blending mode \n');
    fprintf('Move the cursor up and down within the image to \n');
    fprintf('blend between the input images\n\n');
    
    % Generate nodes for the multi-image linear blending
    [rows,cols,~] = size(im{1});
    if XY == 'Y'
        v = round([0:1/(nImages-1):1] * rows); 
    else
        v = round([0:1/(nImages-1):1] * cols); 
    end
    
    % Set up figure and handles to image data
    if exist('figNo','var')
        fig = figure(figNo); clf;
    else
        fig = figure;
    end
    
    S = warning('off');
    imshow(im{ max(1,fix(nImages/2)) }, 'border', 'tight');
    drawnow
    warning(S)
    
%    set(fig, 'NumberTitle', 'off')
    set(fig, 'name', 'CET Image Blender')
    set(fig,'Menubar','none');
    ah = get(fig,'CurrentAxes');
    imHandle = get(ah,'Children'); 
    
    % Set up button down callback
    set(fig,'WindowButtonDownFcn',@wbdcb);
    blending = 0;
    
    % Set up custom pointer
    myPointer = [circularstruct(7) zeros(15,1); zeros(1, 16)];
    myPointer(~myPointer) = NaN;
    
    hspot = 0;
    hold on

%-----------------------------------------------------------------------    
% Window button down callback.  This toggles blending on and off changing the
% cursor appropriately.

function wbdcb(src,evnt)
    if strcmp(get(src,'SelectionType'),'normal')
        if ~blending  % Turn blending on
            blending = 1;
%            set(src,'Pointer','top')
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
            hspot = plot(x, y, '.', 'color', [0 0 0], 'Markersize', 45'); 
            
        end
    end
end 

%-----------------------------------------------------------------------    
% Window button move call back

function wbmcb(src,evnt)
    cp = get(ah,'CurrentPoint');
    y = cp(1,2);
    blend(y)
end

%-----------------------------------------------------------------------
function blend(y)
    
    y = max(0,y); y = min(rows,y);  % clamp y to range 0-rows
    
    % Find y distance from each of the vertices v
    ydist = abs(v - y);
    
    % Find the two closest vertices
    [ydist, ind] = sort(ydist);      
    
    % w1 is the fractional distance from the cursor to the 2nd image
    % relative to the distance between the 1st and 2nd images
    w1 = ydist(2)/(ydist(1)+ydist(2));
    
    % Apply the logistics wighting function to w1 to obtain the desired
    % transition weighting
    w = logisticweighting(w1, B, [0 1 0 1]);
    
    blendim = w*im{ind(1)} + (1-w)*im{ind(2)};
    
    set(imHandle,'CData', blendim);
    
    % See if there is a number at the end of the image file names that is
    % preceeded by a '-' or '_'.  If so assume it represents a scale and
    % display it in the figure header.
    if exist('fname','var')
        [tok1,rem] = strtok(fliplr(basename(fname{ind(1)})),'-'); 
        [tok2,rem] = strtok(fliplr(basename(fname{ind(2)})),'-'); 
        try
            scale1 = eval(fliplr(tok1));
            scale2 = eval(fliplr(tok2)); 
            validScale = 1;
        catch
            validScale = 0;
        end
        
        imtitle = strtok(fname{1},'-');
        
        if validScale
            % Compute a linear estimate of the blended scale.  This will be
            % approximate if the scales vary geometrically but if the total
            % scale range has been covered by numerous images the error should
            % not be too large
            blendscale = ydist(2)/(ydist(1)+ydist(2))*scale1 +...
                ydist(1)/(ydist(1)+ydist(2))*scale2;        
            
            str = sprintf('CET Image Blender    Image: %s  Scale: %4d', ...
                          imtitle, round(blendscale));
        else
            str = sprintf('CET Image Blender    Image: %s', imtitle);
        end
    else
        str = 'CET Image Blender';
    end
    
    set(fig, 'name', str)                
end 

%---------------------------------------------------------------------------
end % of linimix
