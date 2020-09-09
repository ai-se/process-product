function [Xs,Xt] = matchingMetric(train_data,test_data)
dims = size(train_data,1);
dimt = size(test_data,1);
source = train_data;
target = test_data;
pValue = [];
for k=1:dims*dimt
    i = mod(k-1,dims)+1;    % row
    j = floor((k-1)/dims)+1;  % column
    
    s = source(i,:);
    t = target(j,:);
    [~,p] = kstest2(s,t);
    pValue(k) = p;
end
pValue = reshape(pValue,dims,dimt);

temp = pValue;
cutoffValue = 0.05;
temp(pValue<=cutoffValue) = 0;
[~,mi,mj] = bipartite_matching(temp);

if isempty(mi)
    [~,mi,mj] = bipartite_matching(pValue);
end    
Xs = source(mi,:);
Xt = target(mj,:);
end