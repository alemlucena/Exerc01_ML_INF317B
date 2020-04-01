clear all, close all
%% Questão 5 - Item c)
rng(1) % mantem o mesmo sorteio aleatório

d1 = 3;
d2 = 2;
d3 = 0;
d4 = 1;

mu1 = [d1 d2]'; % média classe 1
mu2 = [d3 d4]'; % média classe 2

p1 = 1/2;
p2 = 1/2;

sigma = [1 0.3;0.3 2];

w = inv(sigma)*(mu1-mu2);
x0 = 0.5*(mu1+mu2)-(1/((mu1-mu2)'*inv(sigma)*(mu1-mu2)))*(mu1-mu2)*log(p1/p2);

k = linspace(-10,10,100);

v = (-w(1)*k+w(1)*x0(1)+w(2)*x0(2))/w(2);

% Equação da reta -> x2 = a*x1 + b
a = (-w(1)/w(2));
b = (w(1)*x0(1)+w(2)*x0(2))/w(2);

% Contour plot
x1 = linspace(-10,10,100);
x2 = x1;
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];
y1 = mvnpdf(X,mu1',sigma);
y2 = mvnpdf(X,mu2',sigma);

y1 = reshape(y1,length(x2),length(x1));
y2 = reshape(y2,length(x2),length(x1));

figure,
hold on
contour(x1,x2,y1)
contour(x1,x2,y2)
tmp = plot(k,v,'k--','LineWidth',1.5);
title('Limiar de separação - LDA','FontSize',14);
scatter(mu1(1),mu1(2),300,[0, 0.4470, 0.7410],'.');
scatter(mu2(1),mu2(2),300,[0.8500, 0.3250, 0.0980],'.');
legend(tmp,sprintf('x_2 = %.1fx_1 + %.1f',a,b))
axis([-3 6 -3 6]);
axis square
grid
xlabel('x_1','FontSize',14)
ylabel('x_2','FontSize',14)

%% Simulação N = 1000 pontos item e)

N = 1000;

s1 = mvnrnd(mu1',sigma,round(p1*N))';
s2 = mvnrnd(mu2',sigma,round(p2*N))';

figure
hold on
scatter(s1(1,:),s1(2,:),[],[0, 0.4470, 0.7410],'o');
scatter(s2(1,:),s2(2,:),[],[0.8500, 0.3250, 0.0980],'+');
plot(k,v,'k--','LineWidth',1.5);
scatter(x0(1),x0(2),'k*');
scatter([mu1(1) mu2(1)],[mu1(2) mu2(2)],300,'k','.');
plot([mu1(1) mu2(1)],[mu1(2) mu2(2)],'--','Color',[0.8 0.8 0.8])
title('Simulação Classificação - LDA','FontSize',14);
legend('Classe 1','Classe 2',sprintf('x_2 = %.1fx_1 + %.1f',a,b));
axis([-3 6 -3 6]);
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