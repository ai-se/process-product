function [train_new,Ys,test_new,Yt] = HDP_KS(data,target)
% preprocess target data
[Xt,Yt] = preprocess_source(target);

source_data = data;

% feature selection for source data
% source_data=featureSelection(source_data,source_name,source_metric,selection_method,r);
% preprocess source data
[Xs,Ys] =  preprocess_source(source_data);
   
% matching metric
[train_new,test_new] = matchingMetric(Xs,Xt);
train_new = zscore(train_new,0,2);
test_new = zscore(test_new,0,2);
train_new = train_new';
test_new = test_new';
end