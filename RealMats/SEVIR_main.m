%%% SEVIR_main.m

%%%% main file to run CSS algs on SEVIR model using k
%%%% demonstrates how to generate sens matrix

clear 
close all

%%%%%% Initial Conditions
N = 332.6;
S0 = 285.1;
E0 = 10;
V0 = 0;
I0 = 1;
R0 = 0;

X0 = [S0; E0; V0; I0; R0];

%%%%%% Baseline Parameter Values
beta = 0.8;
alpha = 0.1;
eta = 0.33;
nu = 0.004;
gamma = 0.14;

params = [beta; alpha; nu; eta; gamma];

[t, y] = SEVIR(X0, params); 

h = 1e-10; %%% step size for deriv approx
numpar = length(params);
for par = 1:numpar
    temp = params;
    temp(par) = complex(params(par),h); %%% complex perturbation

    [~,y_cs] = SEVIR(X0,temp);
    y_cs = y_cs(:,4);
    y_cs = y_cs(1:10:end,:);

    S_SEVIR(:,par) = imag(y_cs)/h;
end

tol = 1e-8;
PSS_method = 1:4;
[Unid_SEVIR, ~, crit_SEVIR] = SubsetSelect(S_SEVIR,[],4, PSS_method);