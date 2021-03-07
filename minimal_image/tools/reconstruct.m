function Ihat = reconstruct(bases, w)

Ihat = sum(double(bases.B(:,:,:,w==1)),4);
    
Ihat = Ihat - min(Ihat(:));
Ihat = (Ihat / max(Ihat(:)))*255;

% for i = size(bases.channel_means)
%     Ihat(:,:,i) = Ihat(:,:,i) + bases.channel_means(i) - mean(mean(Ihat(:,:,i)));
% end
