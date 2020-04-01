clear all, close all
%% Quest�o 7
rng(1) % mantem o mesmo sorteio aleat�rio

% Importa datasets
for ii=1:6
    eval(['load ',sprintf('dataset%i.txt',ii),';']);
end

% eval(['data = ',sprintf('dataset%i',ii),';']);

data = dataset4(:,1:2)';
labels = dataset4(:,3);
train_percent = 3/4;

[train_data,train_label,test_data,test_label] = my_holdout(data,labels,train_percent);

%% QDA

s1 = train_data(:,find(train_label==1));
s2 = train_data(:,find(train_label==2));

mu1 = mean(s1,2); % m�dia classe 1
mu2 = mean(s2,2); % m�dia classe 2

p1 = length(s1)/length(train_data);
p2 = length(s2)/length(train_data);

sigma1 = cov(s1');
sigma2 = cov(s2');

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

%surface fronteira decis�o
surf_range = 8;
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
h1 = ezplot(g1==g2,[-15 15]);
set(h1,'color',[0 0 0]);
set(h1,'LineStyle','--')
hold on
sc1 = scatter(s1(1,:),s1(2,:),[],[0, 0.4470, 0.7410],'o');
sc2 = scatter(s2(1,:),s2(2,:),[],[0.8500, 0.3250, 0.0980],'+');
map = [sc1.CData;sc2.CData];
h2 = surf(X,Y,Z);
alpha 0.3
view(2);
colormap(map)
set(h2,'edgecolor','none');
title('Simula��o Classifica��o - QDA','FontSize',14);
legend([sc1 sc2 h1],'Classe 1','Classe 2','Limiar QDA');
axis([-8 8 -8 8]);
pbaspect([1 1 1])
xlabel('x_1','FontSize',14)
ylabel('x_2','FontSize',14)
grid

%% Valida��o

c = test_data;
labels = test_label; % Classe 1 = 1 e Classe 2 = 2;

TP = 0;
FP = 0;
TN = 0;
FN = 0;
N = length(c);
for i=1:N
    
    x = c(:,i);
    
    g1 = x'*W1*x + w1'*x + w10;
    g2 = x'*W2*x + w2'*x + w20;
    
    g = g2-g1;
    
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

[TP FP;FN TN]
CM = [TP/N FP/N;FN/N TN/N];

acc = (TP+TN)/N; % Acur�cia
err = (FP+FN)/N; % Erro