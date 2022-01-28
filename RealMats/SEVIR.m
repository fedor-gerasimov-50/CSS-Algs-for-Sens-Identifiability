%%%  filename: SEVIR.m
%%%  computes numerical solution of ODE system
%%%  Author: Kate Pearce

%%%  Inputs: 
%%%       params: model parameters
%%%           beta: transmission coeff = rate at which susceptible and infectious come into contact * probability of transmission [1/T * diml]
%%%           alpha: reduced fraction of transmission from vaccination (1 - vacc efficacy rate)
%%%           nu: vaccination rate
%%%           eta: rate of progression to infectious after exposure
%%%           gamma: rate of progression through infected stage 
%%%       X0 (optional): initial condition for state vector X = [S; E; V; I; R]

%%% Outputs: 
%%%       t: time vector 
%%%       y: soln vector 

function [t,y] = SEVIR(X0, params)

tfinal = 30; %%% Specify time steps to record the solution (in days)
tspan = 0:0.1:tfinal; 

options = odeset('AbsTol',1e-4); 

frhs = @(t, y)(SEVIRrhs(t, y, params));    %%% anonymous sub-function for ODE solver
 
[t,y] = ode45(frhs, tspan, X0, options); %%% calls numerical solver

%%% sub-function SVIRrhs: creates right hand side of ODE system
    function yprime = SEVIRrhs(t, x, params)
        
        N = 332.6; %%% total population
        
        %%% state variables
        S = x(1);
        E = x(2);
        V = x(3);
        I = x(4);
        
        %%% model parameters
        beta = params(1);
        alpha = params(2);
        nu = params(3);
        eta = params(4);
        gamma = params(5);

        yprime = zeros(5, 1);

        yprime(1) = -beta * I * S/N - nu * S; %%% S
        yprime(2) = beta * I * S/N + alpha * beta * I * V/N - eta*E; %%% E
        yprime(3) = -alpha * beta * I * V/N + nu * S; %%% V
        yprime(4) = eta * E - gamma * I; %%% I
        yprime(5) = gamma * I; %%% R
    end

end


