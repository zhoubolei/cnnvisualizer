% MAP2GEOSOFTTBL Converts MATLAB colourmap to Geosoft .tbl file
%
% Usage:  map2geosofttbl(map, filename)
%
% Arguments:   map - N x 3 rgb colourmap
%         filename - Output filename
%
% This function converts a RGB colourmap to KCMY and writes it out as a .tbl
% file that can be loaded into Geosoft Oasis Montaj
%
% See also: RGB2CMYK

% Peter Kovesi
% Centre for Exploration Targeting
% The University of Western Australia
% peter.kovesi at uwa edu au

% PK October 2012

function map2geosofttbl(map, filename)

    % Convert RGB values in map to CMYK and scale 0 - 255
    cmyk = round(rgb2cmyk(map)*255);
    kcmy = circshift(cmyk, [0 1]); % Shift K to 1st column
    N = size(kcmy,1);
    
    fid = fopen(filename, 'wt');
    
    fprintf(fid, '{ blk cyn mag yel } \n');
    for n = 1:N
      fprintf(fid, '  %03d %03d %03d %03d \n', kcmy(n,:));    
    end
    
    fclose(fid);
    