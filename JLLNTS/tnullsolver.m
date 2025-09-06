function [Z,G] = tnullsolver(X,lambda,maxIter)
%% ---------- Initilization -------- %
%% 参数
[m,n] = size(X);
afa=lambda;%E21
beta=0.01;%tr
% gama=0.001;%|w|
miu = 1e-4;
invmiu=1/miu;
rho = 1.1;
max_miu = 1e8;
tol  = 1e-5;
iter = 0;
%% 矩阵初始化
% 计算样本矩阵的 K 最近邻图
% [idx, ~] = knnsearch(X', X', 'k', k+1);%这个函数需要输入n*d
% % 去除每个样本自身，得到有效的最近邻索引
% idx = idx(:, 2:end);
% % 构建近邻矩阵
% adjacency_matrix = zeros(n);
% for i = 1:n
%     adjacency_matrix(i, idx(i, :)) = 1;
% end
I = eye(n);
A1 = zeros(m,n);
A2 = zeros(m,n);
H = zeros(n,n,2);
T=H;
G=H;
W = H(:,:,1) ;  
Z = H(:,:,2) ;
E = zeros(m,n);
%%    迭代计算
while iter < maxIter
    iter = iter + 1;

    %% update G
    G_old=G;
    H(:,:,1) = W;  
    H(:,:,2) = Z;
    [G,~] = prox_tnn(H+invmiu*T,invmiu);
    %% update W
    W=(2*I+miu*X'*X+miu*I)\(2*I-beta*Z-X'*A1+miu*G(:,:,1)-T(:,:,2));
    W=max(W,0);
    % for ic = 1:n
    %     idx    = 1:n;
    %     idx(ic) = [];
    %     W(ic,idx) = EProjSimplex_new(W(ic,idx));          %
    % end
    %% update Z
    Z_old=Z;
    L1=X-E+invmiu*A2;
    L2=G(:,:,2)-invmiu*T(:,:,1);
    Z=(miu*I+miu*X'*X)\(miu*X'*L1+miu*L2-beta*W);
    for ic = 1:n
        idx    = 1:n;
        idx(ic) = [];
        Z(ic,idx) = EProjSimplex_new(Z(ic,idx));          %
    end
    Z=max(Z,0);
    Z=Z-diag(diag(Z));
    %% update E
    E_old=E;
    M4=X-X*Z-invmiu*A2;
    m4=[];
    for i=1:n
        m4=[m4;norm(M4(:,i))];
    end
    lm=afa*invmiu;
    for i=1:n
        if m4(i)>lm
            E(:,i)=(1-lm/m4(i))*M4(:,i);
        else
            E(:,i)=zeros(m,1);
        end
    end
    %% update others
    A1=A1+miu*X*W;
    A2=A2+miu*(X-X*Z-E);
    T=T+miu*(H-G);
    miu=min(max_miu,rho*miu);
    %%  stop
    diff1 = max(max(abs(H(:,:,2)-G(:,:,2))));
    diff2 = max(max(abs(X-X*Z-E)));
    diff3 = max(max(abs(H(:,:,1)-G(:,:,1))));
    stopC = max([diff2,diff3]);
    LL1 = norm(Z-Z_old,'fro');
    LL4 = norm(G-G_old,'fro');
    LL5 = norm(E-E_old,'fro');
    SLSL = max(max(LL1,LL4),LL5)/norm(X,'fro');
    if miu*SLSL < tol
        miu = min(rho*miu,max_miu);
    end
    if stopC < tol
        break;
    end
end
end
