% driver for CSS

clear
close all
rng default

[t,y] = evaluate_model('Protein');

% N = 128;
% figure(1)
% for i = 1:length(t)
%     plot(y(i,1:N))
%     axis([0 N -2.1 2.2])
%     pause(0.05)
% end

plot(t,y)


% figure(2)
% imagesc(y(:,1:N)');
% colorbar
% 
% figure(3)
% imagesc(y(:,N+1:end)')
% colorbar

