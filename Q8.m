clear all, close all;
%% Questão 8
rng(4) % mantem o mesmo sorteio aleatório

load dataset2.txt

data=dataset2;

N = length(data);
nSim = 10;

train_percent_a = (1/5);  % item a) Caso 1
train_percent_b = (4/5);  % item b) Caso 2

acc_a = [];
err_a = [];
acc_b = [];
err_b = [];
for ii=1:nSim

[train_data_a,train_label_a,test_data_a,test_label_a] = my_holdout(data(:,1:2)',data(:,3),train_percent_a);
[train_data_b,train_label_b,test_data_b,test_label_b] = my_holdout(data(:,1:2)',data(:,3),train_percent_b);

s1_a=train_data_a(:,find(train_label_a==1));
s2_a=train_data_a(:,find(train_label_a==2));
s1_b=train_data_b(:,find(train_label_b==1));
s2_b=train_data_b(:,find(train_label_b==2));

p1_a = length(s1_a)/length(train_data_a);
p2_a = length(s2_a)/length(train_data_a);
p1_b = length(s1_b)/length(train_data_b);
p2_b = length(s2_b)/length(train_data_b);

mu1_a = mean(s1_a,2); % média classe 1 a
mu2_a = mean(s2_a,2); % média classe 2 a
mu1_b = mean(s1_b,2); % média classe 1 a
mu2_b = mean(s2_b,2); % média classe 2 a

sigma_a = cov([s1_a s2_a]');
sigma_b = cov([s1_b s2_b]');

w_a = inv(sigma_a)*(mu1_a-mu2_a);
x0_a = 0.5*(mu1_a+mu2_a)-(1/((mu1_a-mu2_a)'*inv(sigma_a)*(mu1_a-mu2_a)))*(mu1_a-mu2_a)*log(p1_a/p2_a);
w_b = inv(sigma_b)*(mu1_b-mu2_b);
x0_b = 0.5*(mu1_b+mu2_b)-(1/((mu1_b-mu2_b)'*inv(sigma_b)*(mu1_b-mu2_b)))*(mu1_b-mu2_b)*log(p1_b/p2_b);

k = linspace(-10,10,100);
v_a = (-w_a(1)*k+w_a(1)*x0_a(1)+w_a(2)*x0_a(2))/w_a(2);
v_b = (-w_b(1)*k+w_b(1)*x0_b(1)+w_b(2)*x0_b(2))/w_b(2);

% Equação da reta -> y = a*x + b
a_a = (-w_a(1)/w_a(2));
b_a = (w_a(1)*x0_a(1)+w_a(2)*x0_a(2))/w_a(2);
a_b = (-w_b(1)/w_b(2));
b_b = (w_b(1)*x0_b(1)+w_b(2)*x0_b(2))/w_b(2);
%% Validação - Item a)

c_a = test_data_a;
labels_a = test_label_a; % Classe 1 = 1 e Classe 2 = 2

TP = 0;
FP = 0;
TN = 0;
FN = 0;
N = length(c_a);
for i=1:N
    g = a_a*c_a(1,i)+b_a-c_a(2,i);
    if g < 0
        if labels_a(i) == 1
            TP = TP + 1;
        else
            FP = FP +1;
        end
    else
        if labels_a(i) == 2
            TN = TN + 1;
        else
            FN = FN +1;
        end
    end
end

[TP FP;FN TN];
CM_a = [TP/N FP/N;FN/N TN/N];

acc_a = [acc_a,(TP+TN)/N]; % Acurácia
err_a = [err_a,(FP+FN)/N]; % Erro

%% Validação - Item b)

c_b = test_data_b;
labels_b = test_label_b; % Classe 1 = 1 e Classe 2 = 2

TP = 0;
FP = 0;
TN = 0;
FN = 0;
N = length(c_b);
for i=1:N
    g = a_b*c_b(1,i)+b_b-c_b(2,i);
    if g < 0
        if labels_b(i) == 1
            TP = TP + 1;
        else
            FP = FP +1;
        end
    else
        if labels_b(i) == 2
            TN = TN + 1;
        else
            FN = FN +1;
        end
    end
end

[TP FP;FN TN];
CM_b = [TP/N FP/N;FN/N TN/N];

acc_b = [acc_b,(TP+TN)/N]; % Acurácia
err_b = [err_b,(FP+FN)/N]; % Erro

%% Figure plot

%surface fronteira decisão
surf_range = 8;
xx=linspace(-surf_range,surf_range,200);
yy=linspace(-surf_range,surf_range,200);
[X,Y]=meshgrid(xx,yy);

Z1 = X;
Z2 = X;
for i = 1:length(X)
    for j = 1:length(Y)
        c = [X(i,j);Y(i,j)];
        Z1(i,j) = a_a*c(1)+b_a-c(2);
        Z2(i,j) = a_b*c(1)+b_b-c(2);
    end
end
Z1 = sign(Z1);
Z2 = sign(Z2);

figure
hold on
sc1 = scatter(data(data(:,3)==1,1),data(data(:,3)==1,2),[],[0, 0.4470, 0.7410],'o');
sc2 = scatter(data(data(:,3)==2,1),data(data(:,3)==2,2),[],[0.8500, 0.3250, 0.0980],'+');
plot(k,v_a,'b--','LineWidth',1.5);
plot(k,v_b,'r--','LineWidth',1.5);
title(sprintf('Classificação dataset2 LDA - #%i',ii),'FontSize',14);
leg = legend('Classe 1','Classe 2',sprintf('A) x_2 = %.1fx_1 + %.1f',a_a,b_a),sprintf('B) x_2 = %.1fx_1 + %.1f',a_b,b_b));
axis([-4 4 -4 4]);
pbaspect([1 1 1])
grid
xlabel('x_1','FontSize',14)
ylabel('x_2','FontSize',14)

end

% média acurácia e desvio padrão
[mean(acc_a) sqrt(var(acc_a))] % item a)
[mean(acc_b) sqrt(var(acc_b))] % item b)