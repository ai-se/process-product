function measure = performanceMeasure(test_label, predict_label)

% True Positive: the number of defective modules predicted as defective
% False Negative: the number of defective modules predicted as non-defective
% False Positive: the number of non-defective modules predictied as defective
% Ture Nagative: the number of non-defective modules predictied as non-defective

% confusion matrix
%  
%                            defective             non-defective (true class)
%  ------------------------------------------------------
%   predicted   defective        TP                  FP (Type I error)
%    class      ---------------------------------------------
%               non-defective    FN (Type II error)  TN
%  -------------------------------------------------------
% 
% Pd = recall = TP / (TP+FN)
% 
% Pf = FP / (FP+TN)
% 
% preceison = TP / (TP+FP)
% 
% F-measure = (alpha+1)*recall*precision / (recall+alpha*precison)
% 

[X_coordi,Y_coordi,T_thre,AUC] = perfcurve(test_label',predict_label',1);

predict_label(predict_label>0.5) = 1;
predict_label(predict_label<=0.5) = 0;

test_total = length(test_label);
posNum = sum(test_label == 1); % defective
negNum = test_total - posNum;

pos_index = test_label == 1;
pos = test_label(pos_index);
pos_predicted = predict_label(pos_index);
FN = sum(pos ~= pos_predicted); % Type II error

neg_index = test_label == 0; % defective free
neg = test_label(neg_index);
neg_predicted = predict_label(neg_index);
FP = sum(neg ~= neg_predicted);  % Type I error

error = FN+FP;
% FN = error2;
TP = posNum-FN;
% FP = error1;
TN = negNum-FP;

pd_recall = 0;
pf = 0;
precision = 0;
F_measure = 0;

acc = 1.0-error/test_total;

if TP+FN ~= 0
    pd_recall = TP/(TP+FN); % 1.0-error1/posNum;
end

if FP+TN ~= 0
    pf = FP/(FP+TN); % negNum
end

if TP+FP ~= 0
    precision = TP/(TP+FP); % (posNum-error1)/(posNum-error1+error2);
end

if (pd_recall+precision) ~= 0
    F_measure = 2 * pd_recall * precision / (pd_recall+precision);
end

SF1 = 0;
alpha1 = 0.5; % precision over recall
if alpha1*precision+pd_recall ~= 0
    SF1 = (1+alpha1) * pd_recall * precision / (alpha1*precision+pd_recall);
end

SF2 = 0;
alpha2 = 4; % recall over precision
if alpha2*precision+pd_recall ~= 0
    SF2 = (1+alpha2) * pd_recall * precision / (alpha2*precision+pd_recall);
end

MCC = 0;
temp = sqrt((TP+FN)*(TP+FP)*(FN+TN)*(FP+TN));
if temp ~= 0
    MCC = (TP*TN-FP*FN)/temp;
end

if pd_recall+1-pf ~= 0
    g_measure = 2*pd_recall*(1-pf)/(pd_recall+1-pf);
else
    g_measure = 0;
end

GM = sqrt(pd_recall.*(1-pf));

% % balance = 1-sqrt(((0-pf).^2+(1-pd).^2)/2)
bal = 1- ( sqrt((0-pf).^2+(1-pd_recall).^2) ./ sqrt(2) );

measure = [acc, precision, pd_recall, pf, F_measure, SF2, AUC, g_measure, GM, bal, MCC, SF1, TP, FP, FN, TN];
