%%% Parameter Subset Selection Algorithm with PCA B4 (PSS_B4.m)

%%% Determines unidentifiable parameters using parameter subset selection algorithm
%%% with PCA B4 from Jolliffe 1972

%%% Inputs:
    %%% dydq: sensitivity matrix (double, n x p with n >= p)
    %%% eta (optional, double): threshold for info matrix rank; default 1e-8
    %%% k (optional, integer): rank, or number identifiable params 
    
%%% Outputs:
    %%% UnId: unidentifiable parameter indices
    %%% Id: identifiable parameter indices
    %%% c: success criteria for rank revealing fact

function [UnId, Id, c] = PSS_B4(dydq, eta, k)

%% initialize values
UnId = [];

p = size(dydq, 2);
Id = 1:p;

c = [];

%% do an initial QR decomposition: DON'T PIVOT FOR DIRECT ALG COMPARISONS
[Q, R] = qr(dydq, 0); 
[~, SingVals, V] = svd(R, 'econ'); 

if (~exist('k', 'var')) || (isempty(k) == 1)
    if (~exist('eta', 'var')) || (isempty(eta) == 1) || (nargin < 2)
        eta = 1e-8; %% default
    end

    %% do the eta stuff: this will give a k
    SingVals = diag(SingVals);
    ind = find((SingVals./SingVals(1)) > eta); 
    k = length(ind); 

end

%% do the k stuff

if k > 0
    %% compute largest sing vector v_1
    v_1 = V(:,1);
    
    [~,max_ind] = max(abs(v_1));
    
    %% move magnitude largest element to the top
    P_tild = 1:p;
    P_tild([1 max_ind]) = P_tild([max_ind 1]);
    
    
    %% compute QR of R*P_tild
    B = R(:,P_tild);
    [Q_tild, R_tild] = qr(B);
    
    Q = Q*Q_tild;
    R = R_tild;
    P = P_tild;
    
    %% main loop
    for l = 2:k %%% indexing is ok: just won't execute if k <= 1
        
        R_22 = R(l:end,l:end);
        
        [~,~,V_l] = svd(R_22);
        v_l = V_l(:,1);
        
        [~, max_ind] = max(abs(v_l));
        
        P_tild_loop = 1:length(v_l);
        P_tild_loop([1 max_ind]) = P_tild_loop([max_ind 1]);
        
        %%% compute qr decomposition of R_22 * P_tild
        [Q_tild_loop, R_tild_loop] = qr(R_22(:,P_tild_loop));
        
        temp = P_tild_loop+ones(size(P_tild_loop))*(l-1); %%% shift by how many you've done!
        temp_ind = [1:(l-1),temp];

        %%% update everything
        Q = Q*blkdiag(eye(l-1), Q_tild_loop);
        P(1:p) = P(temp_ind);
        R(l:end,l:end) = R_tild_loop;

    end
    
    Id = P(1:k);  
    UnId = P(k+1:end);
    [abs_err, rel_err, cond_S, cond_S1] = SuccessCheck(dydq, P, k, SingVals);
    c = [abs_err, rel_err; cond_S, cond_S1] ;
            
end

end