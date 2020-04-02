clear all, close all, clc;
%% Questão 7
rng(1) % mantem o mesmo sorteio aleatório

% Importa datasets
for ii=1:6
    eval(['load ',sprintf('dataset%i.txt',ii),';']);
end

%% Datasets 1 a 4
for k=1:4
clear X
eval(['X = ',sprintf('dataset%i',k),';']);

data = X(:,1:2)';
labels = X(:,3);
train_percent = 3/4;

nSim = 10;

acc = [];
err = [];
for ii=1:nSim
    
    [train_data,train_label,test_data,test_label] = my_holdout(data,labels,train_percent);
    
    %% QDA
    
    s1 = train_data(:,find(train_label==1));
    s2 = train_data(:,find(train_label==2));
    
    mu1 = mean(s1,2); % média classe 1
    mu2 = mean(s2,2); % média classe 2
    
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
    
    %surface fronteira decisão
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
    
    if 0 % Mudar para 1 para exibir figuras
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
    title('Simulação Classificação - QDA','FontSize',14);
    legend([sc1 sc2 h1],'Classe 1','Classe 2','Limiar QDA');
    axis([-8 8 -8 8]);
    pbaspect([1 1 1])
    xlabel('x_1','FontSize',14)
    ylabel('x_2','FontSize',14)
    grid
    end
    
    %% Validação
    
    c = test_data;
    labels2 = test_label; % Classe 1 = 1 e Classe 2 = 2;
    
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
            if labels2(i) == 1
                TP = TP + 1;
            else
                FP = FP +1;
            end
        else
            if labels2(i) == 2
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

end

%% Datasets 5 e 6
for k=5:6
clear X
eval(['X = ',sprintf('dataset%i',k),';']);

data = X(:,1:2)';
labels = X(:,3);
train_percent = 3/4;

acc = [];
err = [];
for ii=1:nSim
    
    [train_data,train_label,test_data,test_label] = my_holdout(data,labels,train_percent);
    
    %% QDA
    
    s1 = train_data(:,find(train_label==1));
    s2 = train_data(:,find(train_label==2));
    s3 = train_data(:,find(train_label==3));
    
    mu1 = mean(s1,2); % média classe 1
    mu2 = mean(s2,2); % média classe 2
    mu3 = mean(s3,2); % média classe 2
    
    p1 = length(s1)/length(train_data);
    p2 = length(s2)/length(train_data);
    p3 = length(s2)/length(train_data);
    
    sigma1 = cov(s1');
    sigma2 = cov(s2');
    sigma3 = cov(s3');
    
    W1 = -(1/2)*inv(sigma1);
    W2 = -(1/2)*inv(sigma2);
    W3 = -(1/2)*inv(sigma3);
    
    w1 = inv(sigma1)*mu1;
    w2 = inv(sigma2)*mu2;
    w3 = inv(sigma3)*mu3;
    
    w10 = -(1/2)*mu1'*inv(sigma1)*mu1 - (1/2)*log(det(sigma1)) + log(p1);
    w20 = -(1/2)*mu2'*inv(sigma2)*mu2 - (1/2)*log(det(sigma2)) + log(p2);
    w30 = -(1/2)*mu3'*inv(sigma3)*mu3 - (1/2)*log(det(sigma3)) + log(p3);
    
    %symbolic expression plot
    syms v1 v2
    p = [v1 v2]';
    
    g1 = p'*W1*p + w1'*p + w10;
    g2 = p'*W2*p + w2'*p + w20;
    g3 = p'*W3*p + w3'*p + w30;
    
    %surface fronteira decisão
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
            tmp3 = x'*W3*x + w3'*x + w30;
            
            [~,I] = max([tmp1 tmp2 tmp3]);
            Z(i,j) = I;
            
        end
    end
%     Z = sign(Z);
    
    if 0 % Mudar para 1 para exibir figuras
    figure
    hold on
%     h1 = ezplot(g1==g2,[-15 15]);
%     set(h1,'color',[0 0 0]);
%     set(h1,'LineStyle','--')
%     h2 = ezplot(g1==g3,[-15 15]);
%     set(h2,'color',[0 0 0]);
%     set(h2,'LineStyle','--')
%     h3 = ezplot(g2==g3,[-15 15]);
%     set(h3,'color',[0 0 0]);
%     set(h3,'LineStyle','--')
    sc1 = scatter(s1(1,:),s1(2,:),[],[0, 0.4470, 0.7410],'o');
    sc2 = scatter(s2(1,:),s2(2,:),[],[0.8500, 0.3250, 0.0980],'+');
    sc3 = scatter(s3(1,:),s3(2,:),[],[0.9290, 0.6940, 0.1250],'*');
    map = [sc1.CData;sc2.CData;sc3.CData];
    hh = surf(X,Y,Z);
    alpha 0.3
    view(2);
    colormap(map)
    set(hh,'edgecolor','none');
    title('Simulação Classificação - QDA','FontSize',14);
    legend([sc1 sc2 sc3],'Classe 1','Classe 2','Classe 3');
    axis([-8 8 -8 8]);
    pbaspect([1 1 1])
    xlabel('x_1','FontSize',14)
    ylabel('x_2','FontSize',14)
    grid
    end
    
    %% Validação
    
    c = test_data;
    labels2 = test_label; % Classe 1 = 1 e Classe 2 = 2;
    
    TP = [0 0 0];
    N = length(c);
    for i=1:N
        
        x = c(:,i);
        
        g1 = x'*W1*x + w1'*x + w10;
        g2 = x'*W2*x + w2'*x + w20;
        g3 = x'*W3*x + w3'*x + w30;
        
        [~,I] = max([g1 g2 g3]);
        g = I;
       
        if labels2(i) == I
            TP(I) = TP(I)+1;
        end
        
    end
    
    acc = [acc,(sum(TP))/N]; % Acurácia
end

[mean(acc) sqrt(var(acc))]

end