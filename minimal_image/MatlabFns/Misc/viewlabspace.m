% VIEWLABSPACE  Visualisation of L*a*b* space
%
% Usage:    viewlabspace(dL)
%
% Argument:   dL - Optional increment in lightness with each slice of L*a*b*
%                  space. Defaults to 5
%
% Function allows interactive viewing of a sequence of images corresponding to
% different slices of lightness in L*a*b* space.  Lightness varies from 0 to
% 100.  Initially a slice at a lightness of 50 is displayed.
% Pressing 'l' or arrow up/right will increase the lightness by dL.
% Pressing 'd' or arrow down/left will darken by dL.
% Press 'x' to exit.
%
% In addition, the coordinates of red, green, blue, yellow, cyan and
% magenta are plotted within the space.
%
% See also: LABMAP

% To Do:  Should do a version that cuts a vertical plane through the space
% and rotates about the verical greyscale axis.

% Copyright (c) 2013 Peter Kovesi
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

% March 2013
% November 2013  Interactive Lab coordinate feedback from mouse position

function viewlabspace(dL, figNo)
    
    if ~exist('dL', 'var'), dL = 5; end
    if ~exist('figNo', 'var'), figNo = 1; end
    
    % Define some reference colours in rgb
    rgb = [1 0 0
           0 1 0
           0 0 1
           1 1 0
           0 1 1
           1 0 1];
    
    colours = {'red    '
               'green  '
               'blue   '
               'yellow '
               'cyan   '
               'magenta'};
    
    % ... and convert them to lab
    labv = applycform(rgb, makecform('srgb2lab'));

    % Obtain cylindrical coordinates in lab space
    labradius = sqrt(labv(:,2).^2+labv(:,3).^2);
    labtheta = atan2(labv(:,3), labv(:,2));
    
    % Define a*b* grid for image
    scale = 2;
    [a, b] = meshgrid([-127:1/scale:127]);
    [rows,cols] = size(a);
    
    % Scale and offset lab coords to fit image coords
    labc = zeros(size(labv));
    labc(:,1) = round(labv(:,1));
    labc(:,2) = round(scale*labv(:,2) + cols/2);
    labc(:,3) = round(scale*labv(:,3) + rows/2);
    
    % Print out lab values
    labv = round(labv);
    fprintf('\nCoordinates of standard colours in L*a*b* space\n\n');
    for n = 1:length(labv)
        fprintf('%s  L%3d   a %4d  b %4d    angle %4.1f  radius %4d\n',...
                colours{n}, ...
                labv(n,1), labv(n,2), ...
                labv(n,3),  labtheta(n), round(labradius(n)));
    end
    
    fprintf('\n\n')
    
    % Generate axis tick values
    tickval = [-100 -50 0 50 100];
    tickcoords = scale*tickval + cols/2;
    ticklabels = {'-100'; '-50'; '0'; '50'; '100'};

    
    fig = figure(figNo);
    set(fig, 'WindowButtonMotionFcn', @labcoords)
    texth = text(50, 50,'', 'color', [1 1 1]);
    
    fprintf('Place cursor within figure\n');
    fprintf('Press ''l'' to lighten, ''d'' to darken, or use arrow keys\n');
    fprintf('''x'' to exit\n');
    
    ch = 'l';
    L = 50;
    while ch ~= 'x'

        % Build image in lab space
        lab = zeros(rows,cols,3);
        lab(:,:,1) = L;
        lab(:,:,2) = a;
        lab(:,:,3) = b;

        % Generate rgb values from lab
        rgb = applycform(lab, makecform('lab2srgb'));
        
        % Invert to reconstruct the lab values
        lab2 = applycform(rgb, makecform('srgb2lab'));

        % Where the reconstructed lab values differ from the specified values is
        % an indication that we have gone outside of the rgb gamut.  Apply a
        % mask to the rgb values accordingly
        mask = max(abs(lab-lab2),[],3);
        
        for n = 1:3
            rgb(:,:,n) = rgb(:,:,n).*(mask<2);  % tolerance of 1
        end

        figure(figNo), image(rgb), title(sprintf('Lightness %d', L));
        axis square
        % Recreate the text handle in the new image
        texth = text(50, 50,'', 'color', [1 1 1]);
        
        set(gca, 'xtick', tickcoords);
        set(gca, 'ytick', tickcoords);
        set(gca, 'xticklabel', ticklabels);
        set(gca, 'yticklabel', ticklabels);
        xlabel('a*'); ylabel('b*');    
        impixelinfo

        hold on, 
        plot(cols/2, rows/2, 'r+');   % Centre point for reference
        
        % Plot reference colour positions
        for n = 1:length(labc)
            plot(labc(n,2), labc(n,3), 'w+')
            text(labc(n,2), labc(n,3), ...
                 sprintf('   %s\n  %d %d %d  ',colours{n},...
                         labv(n,1), labv(n,2), labv(n,3)),...
                 'color', [1 1 1])
        end
        
        hold off

        % Handle keypresses within the figure
        pause        
        ch = lower(get(gcf,'CurrentCharacter'));
        if ch == 'l' || ch == 29 || ch == 30
           L = min(L + dL, 100);
        elseif ch == 'd' || ch == 28 || ch == 31
           L = max(L - dL, 0);
        end
        
    end
    
    % Clear callback
    set(fig, 'WindowButtonMotionFcn', '')
    
%------------------------------------------------------
% Window button move call back function
function labcoords(src, evnt)
    cp = get(gca,'CurrentPoint');
    x = cp(1,1); y = cp(1,2);

    aval = round((x-(cols/2))/scale);
    bval = round((y-(rows/2))/scale);
    
    set(texth, 'String', sprintf('a %d   b %d', aval, bval));
end


%--------------------------------------------------------
end % of viewlabspace