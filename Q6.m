clear all, close all
%% Questão 6 - Item a)
rng(1) % mantem o mesmo sorteio aleatório

d1 = 3;
d2 = 2; 
d3 = 0;
d4 = 1;

mu1 = [0 0]'; % média classe 1
mu2 = [0 4]'; % média classe 2

p1 = 1/2;
p2 = 1/2;


% Solução analítica
t = linspace(-10,10,100);
tt =  (22*t.^2-23*log(23)+368)/(184);

sigma1 = [20+d1 0;0 1];
sigma2 = [1 0;0 1];

W1 = -(1/2)*inv(sigma1);
W2 = -(1/2)*inv(sigma2);

w1 = inv(sigma1)*mu1;
w2 = inv(sigma2)*mu2;

w10 = -(1/2)*mu1'*inv(sigma1)*mu1 - (1/2)*log(det(sigma1)) + log(p1);
w20 = -(1/2)*mu2'*inv(sigma2)*mu2 - (1/2)*log(det(sigma2)) + log(p2);

%symbolic expression plot
syms v1 v2
p = [v1 v2]';

g1 = p'*W1*p + w1'*p + w10;
g2 = p'*W2*p + w2'*p + w20;

% Contour plot
x = linspace(-10,10,1000);
[X1,X2] = meshgrid(x,x);
X = [X1(:) X2(:)];
y1 = mvnpdf(X,mu1',sigma1);
y2 = mvnpdf(X,mu2',sigma2);

y1 = reshape(y1,length(x),length(x));
y2 = reshape(y2,length(x),length(x));

figure,
% h = ezplot(g1==g2,[-15 15]);
% set(h,'color',[1 0 0]);
% set(h,'LineStyle','--')
hold on
contour(x,x,y1)
contour(x,x,y2)
scatter(mu1(1),mu1(2),300,[0, 0.4470, 0.7410],'.');
scatter(mu2(1),mu2(2),300,[0.8500, 0.3250, 0.0980],'.');
plot(t,tt,'k--')
caxis([0.005 0.14])
title('Limiar de separação','FontSize',14);
axis([-10 10 -8 12]);
axis square
grid
xlabel('x_1','FontSize',14)
ylabel('x_2','FontSize',14)
%% Simulação - Item b)

N = 2000;

s1 = mvnrnd(mu1',sigma1,round(p1*N))';
s2 = mvnrnd(mu2',sigma2,round(p2*N))';

%surface fronteira decisão
surf_range = 12;
xx=linspace(-surf_range,surf_range,200);
yy=linspace(-surf_range,surf_range,200);
[X,Y]=meshgrid(xx,yy);

Z = X;
for i = 1:length(X)
    for j = 1:length(Y)
        x = [X(i,j);Y(i,j)];
        
        tmp1 = x'*W1*x + w1'*x + w10;
        tmp2 = x'*W2*x + w2'*x + w20;
        
        Z(i,j) = tmp2-tmp1;

    end
end
Z = sign(Z);

figure
h = ezplot(g1==g2,[-15 15]);
set(h,'color',[0 0 0]);
set(h,'LineStyle','--')
hold on
sc1 = scatter(s1(1,:),s1(2,:),[],[0, 0.4470, 0.7410],'o');
sc2 = scatter(s2(1,:),s2(2,:),[],[0.8500, 0.3250, 0.0980],'+');
map = [sc1.CData;sc2.CData];
h2 = surf(X,Y,Z);
alpha 0.3
view(2);
colormap(map)
set(h2,'edgecolor','none');
title('Simulação Classificação - QDA','FontSize',14);
legend([sc1 sc2 h],'Classe 1','Classe 2','Limiar QDA');
axis([-10 10 -8 12]);
pbaspect([1 1 1])
grid
xlabel('x_1','FontSize',14)
ylabel('x_2','FontSize',14)

c = [s1 s2];
labels = [zeros(1,round(p1*N)) ones(1,round(p2*N))]; % Classe 1 = 0 e Classe 2 = 1;

TP = 0;
FP = 0;
TN = 0;
FN = 0;
for i=1:N
    
    x = c(:,i);
    
    g1 = x'*W1*x + w1'*x + w10;
    g2 = x'*W2*x + w2'*x + w20;
    
    g = g2-g1;
    
    if g < 0
        if labels(i) == 0
            TP = TP + 1;
        else
            FP = FP +1;
        end
    else
        if labels(i) == 1
            TN = TN + 1;
        else
            FN = FN +1;
        end
    end
end

[TP FP;FN TN]
CM = [TP/N FP/N;FN/N TN/N];

acc = (TP+TN)/N; % Acurácia
err = (FP+FN)/N; % Erro

%% Simulação - Item c)

sigma = eye(2); % variância 1 - Identidade

w = mu1-mu2;
x0 = 0.5*(mu1+mu2);

k = linspace(-10,10,100);

v = (-w(1)*k+w(1)*x0(1)+w(2)*x0(2))/w(2);

% Equação da reta -> x2 = a*x1 + b
a = (-w(1)/w(2));
b = (w(1)*x0(1)+w(2)*x0(2))/w(2);

Z = X;
for i = 1:length(X)
    for j = 1:length(Y)
        c = [X(i,j);Y(i,j)];
        Z(i,j) = a*c(1)+b-c(2);
    end
end
Z = -sign(Z);

figure
hold on
scatter(s1(1,:),s1(2,:),[],[0, 0.4470, 0.7410],'o');
scatter(s2(1,:),s2(2,:),[],[0.8500, 0.3250, 0.0980],'+');
plot(k,v,'k--','LineWidth',1.5);
scatter(x0(1),x0(2),'k*');
h2 = surf(X,Y,Z);
alpha 0.3
view(2);
colormap(map)
set(h2,'edgecolor','none');
title('Simulação Classificação - LDA','FontSize',14);
legend('Classe 1','Classe 2',sprintf('x_2 = %.1fx_1 + %.1f',a,b));
axis([-10 10 -8 12]);
pbaspect([1 1 1])
grid
xlabel('x_1','FontSize',14)
ylabel('x_2','FontSize',14)

c = [s1 s2];
labels = [zeros(1,round(p1*N)) ones(1,round(p2*N))]; % Classe 1 = 0 e Classe 2 = 1;

TP = 0;
FP = 0;
TN = 0;
FN = 0;
for i=1:N
    g = a*c(1,i)+b-c(2,i);
    if g > 0
        if labels(i) == 0
            TP = TP + 1;
        else
            FP = FP +1;
        end
    else
        if labels(i) == 1
            TN = TN + 1;
        else
            FN = FN +1;
        end
    end
end

[TP FP;FN TN]
CM = [TP/N FP/N;FN/N TN/N];

acc = (TP+TN)/N; % Acurácia
err = (FP+FN)/N; % Erro
