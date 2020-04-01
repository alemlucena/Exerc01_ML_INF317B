clear all, close all;
%% Questão 3 - Item a)
rng(1) % mantem o mesmo sorteio aleatório

x = linspace(-10,10,1000);

d2 = 2;

p1 = 1/2;
p2 = 1/2;

var1 = 1;
mu1 = d2;

var2 = 4;
mu2 = d2;

c1 = (1/sqrt(2*pi*var1))*exp(-0.5*(x-mu1).^2);
c2 =  (1/sqrt(2*pi*var2))*exp(-0.5*(x-mu2).^2);

v1 = (2/3)*(3-sqrt(log(64)));
v2 = (2/3)*(3+sqrt(log(64)));% fronteira de decisão - solução analítica

figure

subplot(121),hold on
subplot(121),area(x,c1,'FaceColor',[0, 0.4470, 0.7410],'EdgeColor',[0, 0.4470, 0.7410]);
subplot(121),area(x,c2,'FaceColor',[0.8500, 0.3250, 0.0980],'EdgeColor',[0.8500, 0.3250, 0.0980]);
alpha .5
grid
axis([-2 6 0 0.5])
axis square
legend('Classe 1','Classe 2')
xlabel('x','FontSize',14);
ylabel('p(x|\omega_i)','FontSize',14)

subplot(122),hold on
subplot(122),area(x,p1*c1,'FaceColor',[0, 0.4470, 0.7410],'EdgeColor',[0, 0.4470, 0.7410]);
subplot(122),area(x,p2*c2,'FaceColor',[0.8500, 0.3250, 0.0980],'EdgeColor',[0.8500, 0.3250, 0.0980]);
plot(v1*ones(1,1000),x,'--k');
plot(v2*ones(1,1000),x,'--k');
alpha .5
grid
axis([-2 6 0 0.5])
axis square
legend('Classe 1','Classe 2','Limiar')
xlabel('x','FontSize',14);
ylabel('P(\omega_i)p(x|\omega_i)','FontSize',14)

set(gcf,'Position',[108.2 108.2 1253.6 648])

%% Questão 3 - Item c ) Simulação 1000 objetos

N = 1000;

s1 = sqrt(var1)*randn(1,round(p1*N))+mu1;
s2 = sqrt(var2)*randn(1,round(p2*N))+mu2;

c = [s1 s2];
labels = [zeros(1,round(p1*N)) ones(1,round(p2*N))]; % Classe 1 = 0 e Classe 2 = 1;

TP = 0;
FP = 0;
TN = 0;
FN = 0;
for i=1:N
    if ((c(i) > v1) & (c(i) < v2));
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