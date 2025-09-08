
clear all
clc
clear memory;
warning off;
addpath  D:\PHDCODE\研究生的代码和数据\4.MVC\all\alldata;
addpath  D:\PHDCODE\研究生的代码和数据\4.MVC\all\Ncut_9;
datasetname = { ...
    'AR', ... 1
    'COIL20', ...3
    'ORL', ...9
    'COIL100', ...13 
}; 
for i =[13]%length(datasetname)
name = datasetname{i};
methodname='TNULL';

load(name)
fea = double(fea)';
selected_class = 60;%length(unique(gnd))
% nnClass = length(unique(gnd));     % The number of classes
select_sample = [];
select_gnd    = [];
for i = 1:selected_class
    idx = find(gnd == i);
    idx_sample    = fea(idx,:);
    select_sample = [select_sample;idx_sample];
    select_gnd    = [select_gnd;gnd(idx)];
end
    
fea = select_sample';
fea = fea./repmat(sqrt(sum(fea.^2)),[size(fea,1) 1]);
gnd = select_gnd;
c   = selected_class;
X = fea;
clear fea select_gnd select_sample idx
[m,n] = size(X);
parm_all=[10];%
result_all=[];
for parm=parm_all
    tic;
    [Z,G]= tnullsolver(X,parm,15) ;
    Z_out = Z;
    A = Z_out;
    A = A - diag(diag(A));
    A = (A+A')/2;
    [NcutDiscrete,NcutEigenvectors,NcutEigenvalues] = ncutW(A,c);
    [value,result_label] = max(NcutDiscrete,[],2);
    % result_label=SpectralClustering(Z, c, 3);
   results = getFourMetrics(result_label,gnd);
   elapsed_time = toc; 
   result_all=[result_all;results elapsed_time]
   % save(char([methodname,'_',name,'_',num2str(parm),'_all']), 'results','A','Z');
end
  save(char([methodname,'_',name,'60class_all']), 'result_all','A','Z');
end

