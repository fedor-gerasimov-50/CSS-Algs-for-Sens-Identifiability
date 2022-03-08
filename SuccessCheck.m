%%% filename: SuccessCheck.m 

%%% Inputs: S (matrix, n x p), 
%%%         P (vector, 1 x p)
%%%         k (numerical rank of S)
%%%         Sig (optional; vector, p x 1 singular values of S)
%%%         where submatrix column partition of S*P into [S1 S2] gives
%%%         approximate basis S1 for range(S) 

%%% Outputs: abs_err  (absolute errors in checking basis criteria 2.4 and 2.5)
%%%          rel_err  (relative errors in checking basis criteria 2.4 and 2.5)
%%%          cond_S   (condition number of host matrix)
%%%          cond_S1  (condition number of ident cols)
%%%           


function [abs_err, rel_err, cond_S, cond_S1] = SuccessCheck(S, P, k, SingVals)
   abs_err = zeros(2,1);
   rel_err = zeros(2,1);
   
   %%% determine if singular values provided to function already
   if ~exist('SingVals', 'var') || isempty(SingVals)
       SingVals = svd(S, 0);
   end
   
   SingVals = nonzeros(SingVals);
   %%% check criteria 2.4: |Sig(S)_k - Sig(S1)_k|
   S_perm = S(:,P);

   if k == length(P)
       S1 = S_perm;
       cond_S = cond(S1);
       cond_S1 = cond(S1);

   else
       S1 = S_perm(:,1:k);
       S2 = S_perm(:,k+1:end);
       
       SingVals_S1   = svd(S1, 0);
       SigS1_k = SingVals_S1(k);
       Sig_k   = SingVals(k);
       
       cond_S = cond(S);
       cond_S1 = cond(S1);
    
       abs_err(1) = abs(SigS1_k - Sig_k);
       %rel_err(1) = abs_err(1)/Sig_k; 
       rel_err(1) = SigS1_k/Sig_k; %%% gamma_1
       
       %%% check criteria 2.5 | ||(I-S1*pinv(S1))*S|| - Sig(S)_{k+1}|
       Sig_kp1  = SingVals(k+1);
       X = S1\S2;
       crit_norm = norm(S2-S1*X, 2);
       abs_err(2) = crit_norm; %%% two norm of residual for least squares problem
       rel_err(2) = abs_err(2)/Sig_kp1; %%% gamma_2

   end
   
   
end