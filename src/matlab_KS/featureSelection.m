function [ selected_data ] = featureSelection( DataFile,DataFilename,DataMetric,selection_method,ratio )

dataFilepath= ['..\..\..\DATASET\codeOfProcess\arff30\',DataFilename,'.arff'];
if exist(dataFilepath)
    % load the arff file
    fileReader = javaObject('java.io.FileReader', dataFilepath);
    ds = javaObject('weka.core.Instances', fileReader);
    ds.setClassIndex(ds.numAttributes() - 1);
else
    [xs,ys] =  preprocess_source(DataFile);
    ds=matlab2weka(DataFilename,DataMetric,xs',ys');
end
    
switch(selection_method)
    case 'chi-square'
        eval=javaObject('weka.attributeSelection.ChiSquaredAttributeEval');
    case 'relief-F'
        eval=javaObject('weka.attributeSelection.ReliefFAttributeEval');
    case 'gainratio'
        eval=javaObject('weka.attributeSelection.GainRatioAttributeEval');
    case 'significance attribute evaluation'
        eval=javaObject('weka.attributeSelection.SignificanceAttributeEval');
end

eval.buildEvaluator(ds);
rank=javaObject('weka.attributeSelection.Ranker');
attrIndex=rank.search(eval,ds);
attrIndex = attrIndex+1;
% obtain selected data according to the attrIndex and ratio
attrIndex_num=ceil(size(attrIndex,1)*ratio);
selected_data=DataFile([attrIndex(1:attrIndex_num)',end],:);
end