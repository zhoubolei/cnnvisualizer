% SINERAMP   Generates sine on a ramp colourmap test image
%
% The test image consists of a sine wave superimposed on a ramp function The
% amplitude of the sine wave is modulated from its full value at the top of the
% image to 0 at the bottom. 
%
% The image is useful for evaluating the effectiveness of different colourmaps.
% Ideally the sine wave pattern should be equally discernible over the full
% range of the colourmap.  In practice many colourmaps have uneven perceptual
% contrast over their range and often include 'flat spots' of no perceptual
% contrast.
%
% Usage: im = sineramp(sze, amp, wavelen, p)
%
% Arguments:     sze - [rows cols] specifying size of test image.  If a
%                      single value is supplied the image is square.
%                amp - Amplitude of sine wave.
%            wavelen - Wavelength of sine wave in pixels.
%                  p - Power to which the linear attenuation of amplitude, 
%                      from top to bottom, is raised.  For no attenuation use
%                      p = 0. For contrast sensitivity experiments use larger
%                      values of p.  The default value is 2.
% 
% The ramp function that the sine wave is superimposed on is sized so that the
% overall greyscale range of the test image is 0-255 (nominally).  Thus using a
% large sine wave amplitude will result in a reduced ramp function being used.
%
% To start with try
%  >> im = sineramp([256 512], 10, 10, 2);
%
% View it under 'gray' then try the 'jet', 'hsv', 'hot' etc colourmaps.  The
% results may cause you some concern!
%
% See also: CHIRPLIN, CHIRPEXP

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

% July 2013

function im = sineramp(sze, amp, wavelen, p)
    
    if length(sze) == 1
        rows = sze; cols = sze;
    elseif length(sze) == 2
        rows = sze(1); cols = sze(2);
    else
        error('size must be a 1 or 2 element vector');
    end

    if ~exist('p', 'var'), p = 2; end

    % Sine wave
    x = 0:cols-1;
    fx = amp*sin( 1/wavelen * 2*pi*x - pi/2);
    
    % Vertical modulating function
    A = ([(rows-1):-1:0]/(rows-1)).^p;
    im = A'*fx;
    
    % Add ramp
    ramp = meshgrid(0:(cols-1), 1:rows)/(cols-1);
    im = im + ramp*(255 - 2*amp) + amp;
    