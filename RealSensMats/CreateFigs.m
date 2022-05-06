% Figures from 'Robust Parameter Identifiability Analysis'
% Pearce et al. 2022

clear
close all
load SHIPSrlzns.mat

A = crit1_rel(1,:)';
B = crit1_rel(2,:)';
C = crit1_rel(3,:)';
D = crit1_rel(4,:)';
group = [    ones(size(A));
         2 * ones(size(B));
         3 * ones(size(C));
         4 * ones(size(D))];
figure(1)
bp = boxplot([A; B; C; D],group);
set(gca,'FontSize',24);
set(bp,'LineWidth', 2);
xticklabels({'Alg 4.1','Alg 4.2','Alg 4.3','Alg 4.4'})

A2 = crit2_rel(1,:)';
B2 = crit2_rel(2,:)';
C2 = crit2_rel(3,:)';
D2 = crit2_rel(4,:)';
group2 = [    ones(size(A2));
         2 * ones(size(B2));
         3 * ones(size(C2));
         4 * ones(size(D2))];
figure(2)
bp2 = boxplot([A2; B2; C2; D2],group2);
set(gca,'FontSize',24);
set(bp2,'LineWidth', 2);
xticklabels({'Alg 4.1','Alg 4.2','Alg 4.3','Alg 4.4'})

A3 = crit3_cnd(1,:)';
B3 = crit3_cnd(2,:)';
C3 = crit3_cnd(3,:)';
D3 = crit3_cnd(4,:)';
group3 = [    ones(size(A3));
         2 * ones(size(B3));
         3 * ones(size(C3));
         4 * ones(size(D3))];
figure(3)
bp3 = boxplot([A3; B3; C3; D3],group3);
ax = gca;
ax.FontSize = 24;
ax.YAxis.Scale = "log";
axis([0.5 4.5 10^(-12.05) 10^(-11)])
set(bp3,'LineWidth', 2);
xticklabels({'Alg 4.1','Alg 4.2','Alg 4.3','Alg 4.4'})
