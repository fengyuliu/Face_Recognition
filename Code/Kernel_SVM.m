% #3 Kernel SVM: Implement the Kernel SVM classifier by solving the dual optimization problem 
%                (or using stochastic gradient descent on the cost function). Use the Radial Basis Function(RBF)
%                kernel and the polynomial kernel. (SMO method)
%                neutral vs. facial expression classification
clc; clear; close all;

s = input('Please input a number: #1 Radial Basis Function (RBF) kernel, #2 polynomial kernel: ');
if ismember(s,[1,2])~=1
    disp('error.')
    return
end

load('./DATA/data.mat');
face_r = reshape(face,24*21,600); %reshape the face
Ntrain = 50;
Ntest = 100-Ntrain;
Dtrain = face_r(:,[3*(1:Ntrain)-2,3*(1:Ntrain)-1]); %use 1 ~ Ntrain pictures in each class for training
Ltrain = [-1*ones(1,Ntrain),1*ones(1,Ntrain)]; %Label of training set -1:neutral face +1: face expression
Dtest = face_r(:,[3*((Ntrain+1):100)-2,3*((Ntrain+1):100)-1]); %use 100-Ntest ~ 100 pictures in each class for testing
Ltest = [-1*ones(1,Ntest),1*ones(1,Ntest)]; %Label of testing set
Nc = 2; %number of classes
Nd = 24*21; %dimension

% %% Cross-validation
% load('./DATA/data.mat');
% face_r = reshape(face,24*21,600); %reshape the face
% Ntrain = 20;
% Ntest = 80;
% D1 = face_r(:,[3*(1:Ntrain)-2,3*(1:Ntrain)-1]);
% D2 = face_r(:,[3*((Ntrain+1):2*Ntrain)-2,3*((Ntrain+1):2*Ntrain)-1]);
% D3 = face_r(:,[3*((2*Ntrain+1):3*Ntrain)-2,3*((2*Ntrain+1):3*Ntrain)-1]);
% D4 = face_r(:,[3*((3*Ntrain+1):4*Ntrain)-2,3*((3*Ntrain+1):4*Ntrain)-1]);
% D5 = face_r(:,[3*((4*Ntrain+1):5*Ntrain)-2,3*((4*Ntrain+1):5*Ntrain)-1]);
% 
% Dtrain = D5;
% Ltrain = [-1*ones(1,Ntrain),1*ones(1,Ntrain)]; %Label of training set -1:neutral face +1: face expression
% Dtest = [D4,D1,D2,D3];
% Ltest = [Ltrain,Ltrain,Ltrain,Ltrain];
% Nc = 2; %number of classes
% Nd = 24*21; %dimension






% tt = Dtrain;
% ll = Ltrain;
% nn = Ntrain;
% Dtrain = Dtest;
% Ltrain = Ltest;
% Ntrain = Ntest;
% Dtest = tt;
% Ltest = ll;
% Ntest = nn;


x = 1.2; % parameter for the kernel: sigma (about 2.5) for #1, r (about 1.2) for #2 
C = 10000; %for separable data, set it to a very large number
alpha = zeros(1,Nc*Ntrain);
b = 0;
k=0;

while k < 1000
    k = k+1;
    for i = 1:Nc*Ntrain
        ui = alpha.*Ltrain*Kernel(Dtrain,Dtrain(:,i),x,s) + b;
        Ei = ui - Ltrain(i);
        if (Ltrain(i)*ui<1 && alpha(i)<C)||(Ltrain(i)*ui>1 && alpha(i)>0)
            j = randi(Nc*Ntrain-1);  
            if j >= i
                j = j+1;
            end
            uj =  alpha.*Ltrain*Kernel(Dtrain,Dtrain(:,j),x,s) + b;
            Ej =  uj - Ltrain(j);
            if Ltrain(i) == Ltrain(j)
                L = max(0, alpha(i) + alpha(j) -C);
                H = min(C, alpha(i) + alpha(j));
            else
                L = max(0, alpha(j) - alpha(i));
                H = min(C, C + alpha(j) - alpha(i));
            end
            if L==H
                continue;
            end
            alphaio = alpha(i); %save the old number
            alphajo = alpha(j);
            eta = 2*Kernel(Dtrain(:,i),Dtrain(:,j),x,s)-Kernel(Dtrain(:,i),Dtrain(:,i),x,s)-Kernel(Dtrain(:,j),Dtrain(:,j),x,s);
            alpha(j) = alpha(j) - Ltrain(j)*(Ei-Ej)/eta;
            if alpha(j) > H
                alpha(j) = H;
            elseif alpha(j) < L
                    alpha(j) = L;
            end
            if abs(alpha(j)-alphajo)<10^(-5) %the difference is small enough so it doesn't need to be changed
                continue;
            end
            alpha(i) = alpha(i) + Ltrain(i)*Ltrain(j)*(alphajo-alpha(j));
            
            b1 = b-Ei-Ltrain(i)*(alpha(i)-alphaio)*Kernel(Dtrain(:,i),Dtrain(:,i),x,s)-Ltrain(j)*(alpha(j)-alphajo)*Kernel(Dtrain(:,i),Dtrain(:,j),x,s);
            b2 = b-Ej-Ltrain(i)*(alpha(i)-alphaio)*Kernel(Dtrain(:,j),Dtrain(:,j),x,s)-Ltrain(j)*(alpha(j)-alphajo)*Kernel(Dtrain(:,j),Dtrain(:,j),x,s);
            
            if alpha(i)>=0 && alpha(i)<=C
                b = b1;
            elseif alpha(j)>=0 && alpha(j)<=C
                b = b2;
            else
                b = (b1+b2)/2;
            end
            k = 0; %if alpha is changed, set k to 0
        end
    end
end

result = (alpha.*Ltrain)*Kernel(Dtrain,Dtest,x,s) + b;
result = sign(result)-Ltest;
accuracy = sum(sum(result==0))/Nc/Ntest;
