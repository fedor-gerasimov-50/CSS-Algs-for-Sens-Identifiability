% Column Subset Selection Algorithm with Alg 4.1 B1 (PSS_B1.m)

% Determines unidentifiable parameters using 
% CSS Algorithm 4.1 from 'Robust Parameter Identifiability Analysis'
% (Pearce et al. 2022)

% Written by Kate Pearce

function [UnId, Id, c] = PSS_B1(dydq, eta, k)
% Inputs:
    % dydq: sensitivity matrix (double, n x p with n >= p)
    % eta (optional, double): threshold for info matrix rank; default 1e-8
    % k (optional, integer): rank, or number identifiable params   
% Outputs:
    % UnId: unidentifiable parameter indices
    % Id: identifiable parameter indices
    % c: success criteria for rank revealing fact

% initialize values
UnId = [];
p = size(dydq, 2);
Id = 1:p;
c = [];

% do an initial QR decomposition
[Q, R] = qr(dydq, 0); 
[~, SingVals, V] = svd(R, 'econ'); 

if (~exist('k', 'var')) || (isempty(k) == 1)
    if (~exist('eta', 'var')) || (isempty(eta) == 1) || (nargin < 2)
        eta = 1e-8; % default
    end

    % if eta provided: find k
    SingVals = diag(SingVals);
    ind = find((SingVals./SingVals(1)) > eta); 
    k = length(ind); 

end

% using k = rank(S1)
num_id = k;
num_unid = p - num_id; 

if num_unid > 0
    % compute smallest sing vector v_p
    v_p = V(:,p);
    
    [~,max_ind] = max(abs(v_p));
    
    % move magnitude largest element to the bottom 
    P_tild = 1:p;
    P_tild([max_ind end]) = P_tild([end max_ind]);
    
    % compute QR of R*P_tild
    B = R(:,P_tild);
    [Q_tild, R_tild] = qr(B);
    
    Q = Q*Q_tild;
    R = R_tild;
    P = P_tild;
    
    % main loop
    for iter = 1:num_unid-1
        
        l = p - iter;
        R_11 = R(1:l,1:l);
        R_12 = R(1:l,(l+1):end);
        R_22 = R((l+1):end,(l+1):end);
        
        [~,~,V_l] = svd(R_11);
        v_l = V_l(:,end);
        
        [~, max_ind] = max(abs(v_l));
        
        P_tild_loop = 1:length(v_l);
        P_tild_loop([end max_ind]) = P_tild_loop([max_ind end]);
        
        R_11P = R_11(:,P_tild_loop);
        
        [Q_tild_smlr, R_11_tild] = qr(R_11P);
        
        
        %% update Q, P, and R
        Q = Q*blkdiag(Q_tild_smlr, eye(p-l));
        P(1:l) = P(P_tild_loop);
        R = blkdiag(R_11_tild, R_22);
        R(1:l,(l+1):p) = Q_tild_smlr'*R_12;
    end

    Id = P(1:num_id);
    UnId = P(num_id+1:end);
    SingVals = nonzeros(SingVals);
    [abs_err, rel_err, cond_S, cond_S1] = SuccessCheck(dydq, P, num_id, SingVals);
    c = [abs_err, rel_err; cond_S, cond_S1] ;
   
end

end