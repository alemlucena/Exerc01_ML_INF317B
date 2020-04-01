clear all, close all;
%% Questão 8
rng(1) % mantem o mesmo sorteio aleatório

load dataset2.txt

X=dataset2;

N = length(X);
nSim = 10;

% train_percent = (1/5);     % item a) Caso 1
train_percent = (4/5);  % item b) Caso 2

acc = [];
err = [];
for ii=1:nSim

[train_data,train_label,test_data,test_label] = my_holdout(X(:,1:2)',X(:,3),train_percent);

s1=train_data(:,find(train_label==1));
s2=train_data(:,find(train_label==2));

p1 = length(s1)/length(train_data);
p2 = length(s2)/length(train_data);

mu1 = mean(s1,2); % média classe 1
mu2 = mean(s2,2); % média classe 2

sigma = cov([s1 s2]');

w = inv(sigma)*(mu1-mu2);
x0 = 0.5*(mu1+mu2)-(1/((mu1-mu2)'*inv(sigma)*(mu1-mu2)))*(mu1-mu2)*log(p1/p2);

k = linspace(-10,10,100);
v = (-w(1)*k+w(1)*x0(1)+w(2)*x0(2))/w(2);

% Equação da reta -> y = a*x + b
a = (-w(1)/w(2));
b = (w(1)*x0(1)+w(2)*x0(2))/w(2);

%% Validação

c = test_data;
labels = test_label; % Classe 1 = 1 e Classe 2 = 2;

TP = 0;
FP = 0;
TN = 0;
FN = 0;
N = length(c);
for i=1:N
    g = a*c(1,i)+b-c(2,i);
    if g < 0
        if labels(i) == 1
            TP = TP + 1;
        else
            FP = FP +1;
        end
    else
        if labels(i) == 2
            TN = TN + 1;
        else
            FN = FN +1;
        end
    end
end

[TP FP;FN TN];
CM = [TP/N FP/N;FN/N TN/N];

acc = [acc,(TP+TN)/N]; % Acurácia
err = [err,(FP+FN)/N]; % Erro
end

[mean(acc) sqrt(var(acc))]
% [mean(err) var(err)]

%% Figure plot

%surface fronteira decisão
surf_range = 8;
xx=linspace(-surf_range,surf_range,200);
yy=linspace(-surf_range,surf_range,200);
[X,Y]=meshgrid(xx,yy);

Z = X;
for i = 1:length(X)
    for j = 1:length(Y)
        c = [X(i,j);Y(i,j)];
        Z(i,j) = a*c(1)+b-c(2);
    end
end
Z = sign(Z);

figure
hold on
sc1 = scatter(s1(1,:),s1(2,:),[],[0, 0.4470, 0.7410],'o');
sc2 = scatter(s2(1,:),s2(2,:),[],[0.8500, 0.3250, 0.0980],'+');
plot(k,v,'k--','LineWidth',1.5);
scatter([mu1(1) mu2(1)],[mu1(2) mu2(2)],300,'k','.');
plot([mu1(1) mu2(1)],[mu1(2) mu2(2)],'--','Color',[0.8 0.8 0.8])
scatter(x0(1),x0(2),'k*');
map = [sc1.CData;sc2.CData];
h2 = surf(X,Y,Z);
alpha 0.3
view(2);
colormap(map)
set(h2,'edgecolor','none');
title('Simulação Classificação - LDA','FontSize',14);
leg = legend('Classe 1','Classe 2',sprintf('x_2 = %.1fx_1 + %.1f',a,b),'\mu_i');
axis([-4 4 -4 4]);
pbaspect([1 1 1])
grid
xlabel('x_1','FontSize',14)
ylabel('x_2','FontSize',14)