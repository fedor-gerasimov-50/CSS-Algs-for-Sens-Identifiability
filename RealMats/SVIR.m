%%%  filename: SVIR.m
%%%  computes numerical solution of ODE system
%%%  Author: Kate Pearce

function [t,y] = SVIR(X0, params)


%%%  Inputs: 
%%%       X0: initial condition for state vector X = [S; V; I; R]
%%%       params: model parameters
%%%           beta: transmission coeff = rate at which susceptible and infectious come into contact * probability of transmission [1/T * diml]
%%%           alpha: reduced fraction of transmission from vaccination (1 - vacc efficacy rate)
%%%           nu: vaccination rate
%%%           gamma: rate of progression through infected stage 

%%% Outputs: 
%%%       t: time vector 
%%%       y: soln vector 

tfinal = 30; %%% Specify time steps to record the solution (in days)
tspan = 0:0.1:tfinal; 

options = odeset('AbsTol',1e-4); 

frhs = @(t, y)(SVIRrhs(t, y, params));    %%% anonymous sub-function for ODE solver
 
[t,y] = ode45(frhs, tspan, X0, options); %%% calls numerical solver

%%% sub-function SVIRrhs: creates right hand side of ODE system
    function yprime = SVIRrhs(t, x, params)
        
        N = 332.6; %%% total population
        
        %%% model parameters
        beta = params(1);
        alpha = params(2);
        nu = params(3);
        gamma = params(4);
        
        S = x(1);
        V = x(2);
        I = x(3);

        yprime = zeros(4, 1);

        yprime(1) = -beta * S * I/N - nu * S; %%% S
        yprime(2) = -alpha * beta * I * V/N + nu * S; %%% V
        yprime(3) = beta * S * I/N + alpha * beta * I * V/N - gamma * I; %%% I
        yprime(4) = gamma * I;
        
    end

end


