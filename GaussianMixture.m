load('GaussianMixtureData.mat')

figure(2); hist(Y);
title('Histogram of the observation of Dataset-1');
xlabel ('Dataset of distribution');
ylabel ('Frequency of Gaussian mixture')

% Initialize
a1(1) = 1/3; mu1(1) = 0.1; sigma1(1) = 1.1;
a2(1) = 1-a1(1); mu2(1) = 5.1; sigma2(1) = 0.5;

T = 10;
n = length(Y);
LogObs = zeros(1,T+1);
LogObs(1) = sum(log(a1(1)*normpdf(Y,mu1(1),sigma1(1)) + a2(1)*normpdf(Y,mu2(1),sigma2(1))));

% EM algorithm
for m = 1:T
    for i = 1:n
        %f1 = normpdf(Y(i),mu1(m), sigma1(m));
        f1 = 1/(sqrt(2*pi)*sigma1(m)^2)*exp(-(Y(i)-mu1(m)).^2/(2*sigma1(m)^2));
        %f2 = normpdf(Y(i),mu2(m), sigma2(m));
        f2 = 1/(sqrt(2*pi)*sigma2(m)^2)*exp(-(Y(i)-mu2(m)).^2/(2*sigma2(m)^2));
        p1(i) = (a1(m)*f1) / (a1(m)*f1 + a2(m)*f2);
        p2(i) = 1 - p1(i);
    end
    
    a1(m+1) = sum(p1)/n;
    a2(m+1) = 1 - a1(m+1);
    
    mu1(m+1) = sum(Y.*p1)/sum(p1);
    mu2(m+1) = sum(Y.*p2) / sum(p2);
    
    sigma1(m+1) = sqrt(sum((Y - mu1(m+1)).^2.*p1) / sum(p1));
    sigma2(m+1) = sqrt(sum((Y - mu2(m+1)).^2.*p2) / sum(p2));
    
    LogObs(m+1) = sum(log(a1(m+1)*normpdf(Y,mu1(m+1),sigma1(m+1)) + a2(m+1)*normpdf(Y,mu2(m+1),sigma2(m+1))));
    %LogObs(m+1) = sum(log(a1(m+1)*1/(sqrt(2*pi)*sigma1(m+1)^2 )*exp(-(Y-mu1(m+1) ).^2/(2*sigma1(m+1)^2))+a2(m+1)*1/(sqrt(2*pi())*sigma2(m+1)^2)*exp(-(Y-mu2(m+1)).^2/(2*sigma2(m+1)^2))));
    
end

disp('    a1    u1        S1^2    a2    Mu2        S2^2')
disp([a1(end),mu1(end),sigma1(end)^2,a2(end),mu2(end),sigma2(end)^2])

figure(3);
plot(1:T+1,LogObs)
title('Log-Likelihood Function of Gaussian Distribution');
