function [train_new,train_label,test_new,test_label] = CTKCCA(data, target_data)
% cost-sentitive transfer KCCA (one source to one target)

source_data=data;
[train_data,train_label]=normN2_source(source_data);
% [~,~,test_data,test_label]=normN2_target(target_data,ridx,ratio);
[test_data,test_label]=normN2_source(target_data)%,ridx,ratio);
%  disp(size(train_data))
% disp(size(test_data))

lrank = 70;  % number of components in incomplete Cholesky decompose
reg = 1E-5; % regularization

d1 = pdist(train_data');
sigma1 = mean(d1);
d2 = pdist(test_data');
sigma2 = mean(d2);

kernel1 = {'gauss',1/sigma1};   % kernel type and kernel parameter for data set 1
kernel2 = {'gauss',1/sigma2};   % kernel type and kernel parameter for data set 2
[Ks,Kt] = conKernelMatrix(train_data',test_data',kernel1,kernel2,lrank);

temp = train_label;
temp(temp==0) = 2;
def_n = length(find(temp==1));
nodef_n = length(find(temp==2));
c = 2; % classes
cost(1) = 1; 
% cost(2) = 1;
cost(2) = nodef_n/def_n;
Css = 0;
Cst = 0;
for i=1:c
    idx = temp == i;
    ksi = Ks(idx,:);
    Css = Css+cost(i)*(ksi'*ksi)+reg*eye(size(ksi,2));
    
    dist = pdist2(ksi,Kt);
    dist = exp(-dist);
    Cst = Cst+cost(i)*ksi'*dist*Kt;
end

Ctt = Kt'*Kt+reg*eye(size(Kt,2));

[Ws,Wt,~] = eigDecomposition(Css,Ctt,Cst);

dim = ceil(size(train_data,1)*0.15); % tune the projected dimension 

Wxx = Ws(:,1:dim);
Wyy = Wt(:,1:dim);
train_new = Ks*Wxx;
test_new = Kt*Wyy;
end
