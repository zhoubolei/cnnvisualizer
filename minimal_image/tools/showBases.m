function showBases(bases)

Bdisplay = 1.4*bases.B+128;
Bdisplay = uint8(Bdisplay);
Bdisplay(1:2,:,:,:)=255;
Bdisplay(:, 1:2,:,:) = 255;

figure
montage(Bdisplay)

