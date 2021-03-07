function Bcontr = getInfoFromBases(B,img)

for i = 1:size(B,4)
    dif_im = mean(B(:,:,:,i),3);
    Bcontr.areaperc(i) = sum(sum(abs(dif_im)>0))/(size(B,1) * size(B,2));
    Bcontr.pixelchange(i) = sum(sum(abs(dif_im)))/(Bcontr.areaperc(i)*(size(B,1) * size(B,2))); 
    Bcontr.globalpixelchange(i) = sum(sum(abs(dif_im)))/(size(B,1) * size(B,2)); 
end
    


