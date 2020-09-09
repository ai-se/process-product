function [Xtl,Ytl,Xtu,Ytu] = normN2_target(target,idx,ratio)
% spliting target data into training target data (Xtl,Ytl) 
% and test target data (Xtu,Ytu)
Xt = target(1:end-1,:);
Yt = target(end,:);
Yt(Yt>1) = 1;

posind = find(Yt == 1); % defective
negind = find(Yt == 0); % non-defectuve
temp1 = Xt(:,posind);
temp2 = Xt(:,negind);
Xt = [temp1, temp2];
Yt = [ones(1,size(temp1,2)), zeros(1,size(temp2,2))];

% split training set and test set
trIdxPos = idx(1:ceil(ratio*length(posind)));
teIdxPos = setdiff(idx(1:length(posind)),trIdxPos);
trIdxNeg = idx(length(posind)+1:length(posind)+ceil(ratio*length(negind)));
teIdxNeg = setdiff(idx(length(posind)+1:end),trIdxNeg);

trIdx = [trIdxPos,trIdxNeg];
teIdx = [teIdxPos,teIdxNeg];
Xtu=Xt(:,teIdx);
Ytu=Yt(:,teIdx);
Xtl=Xt(:,trIdx);
Ytl=Yt(:,trIdx);
clear posind negind temp1 temp2
% normalization
Xtl = zscore(Xtl,0,2);
Xtu = zscore(Xtu,0,2);
end