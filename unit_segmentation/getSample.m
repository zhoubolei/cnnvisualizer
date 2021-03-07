unitID = 49;

featureSet_unit = squeeze(featureStack(:,:,unitID,:));
featureSet_vectorized = reshape(featureSet_unit,[13*13 208754]);
maxValue = max(featureSet_vectorized);
[value_sorted,IDX_sorted ] = sort(maxValue,'descend');

for i=1:5
    curImg = imread(imageList{IDX_sorted(i)});
    curImg = imresize(curImg,[256 256]);
    featureMap = squeeze(featureStack(:,:,unitID,IDX_sorted(i)));
    featureMap = permute(featureMap,[2 1]);
    imwrite(curImg,['data/unitID' num2str(unitID) '_img' num2str(i) '.jpg']);
    save(['data/unitID' num2str(unitID) '_feature' num2str(i) '.mat'],'featureMap');
end