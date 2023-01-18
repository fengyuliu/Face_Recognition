% #4 Boosted SVM: Implement the AdaBoost algorithm for the class of linear SVMs. Report the improvement of the 
%                 boosted classifiers with respect to the iterations of AdaBoost, and compare it with the performance of 
%                 the Kernel SVMs.
%                 neutral vs. facial expression classification
clc; clear; close all;

s = input('Please input the number of iterations you want: ');

load('./DATA/data.mat');
face_r = reshape(face,24*21,600); %reshape the face
Ntrain = 50; % the number of training data in each class
Ntest = 100-Ntrain; % the number of testing data in each class
Dtrain = face_r(:,[3*(1:Ntrain)-2,3*(1:Ntrain)-1]); %use 1 ~ Ntrain pictures in each class for training
Ltrain = [-1*ones(1,Ntrain),1*ones(1,Ntrain)]; %Label of training set -1:neutral face +1: face expression
Dtest = face_r(:,[3*((Ntrain+1):100)-2,3*((Ntrain+1):100)-1]); %use 100-Ntest ~ 100 pictures in each class for testing
Ltest = [-1*ones(1,Ntest),1*ones(1,Ntest)]; %Label of testing set
Nc = 2; %number of classes
Nd = 24*21; %dimension

% tt = Dtrain;
% ll = Ltrain;
% nn = Ntrain;
% Dtrain = Dtest;
% Ltrain = Ltest;
% Ntrain = Ntest;
% Dtest = tt;
% Ltest = ll;
% Ntest = nn;

%parameters for the first step
Ntrain_w = 10; %number of training points used for SVM in each class, shouuld be large enough in case producing bad classifiers
w = ones(1,Nc*Ntrain)/Nc/Ntrain;
P = w/sum(w);
Dtrain_w = Dtrain(:,[(1:Ntrain_w),((Ntrain+1):(Ntrain+Ntrain_w))]);
Ltrain_w = Ltrain([(1:Ntrain_w),((Ntrain+1):(Ntrain+Ntrain_w))]);


%get a weak classifier%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a = zeros(1,s); 
accuracy = zeros(1,s); %accuracy for testing data
accuracy_t = zeros(1,s); %accuracy fot training data
alpha_r = zeros(s,Nc*Ntrain_w);
Ltrain_r = zeros(s,Nc*Ntrain_w);
Dtrain_r = zeros(Nd,Nc*Ntrain_w,s);
b_r = zeros(1,s);
epsilon = zeros(1,s);
LL = zeros(Nc*Ntrain_w,s);%training data used for each iteration. They will lead to different weak classifers.

for it = 1:s

C = 10000; %for separable data, set it to a very large number
alpha = zeros(1,Nc*Ntrain_w);
b = 0;
k=0;
while k < 1000
    k = k+1;
    for i = 1:Nc*Ntrain_w
        ui = alpha.*Ltrain_w*(Dtrain_w.'*Dtrain_w(:,i)) + b;
        Ei = ui-Ltrain_w(i);
        if (Ltrain_w(i)*ui<1 && alpha(i)<C)||(Ltrain_w(i)*ui>1 && alpha(i)>0)
            j = randi(Nc*Ntrain_w-1);  
            if j >= i
                j = j+1;
            end
            uj =  alpha.*Ltrain_w*(Dtrain_w.'*Dtrain_w(:,j)) + b;
            Ej =  uj-Ltrain_w(j);
            if Ltrain_w(i) == Ltrain_w(j)
                L = max(0, alpha(i)+alpha(j)-C);
                H = min(C, alpha(i)+alpha(j));
            else
                L = max(0, alpha(j) - alpha(i));
                H = min(C, C + alpha(j) - alpha(i));
            end
            if L==H
                continue;
            end
            alphaio = alpha(i); %save the old number
            alphajo = alpha(j);
            eta = 2*(Dtrain_w(:,i).'*Dtrain_w(:,j))-(Dtrain_w(:,i).'*Dtrain_w(:,i))-(Dtrain_w(:,j).'*Dtrain_w(:,j));
            alpha(j) = alpha(j) - Ltrain_w(j)*(Ei-Ej)/eta;
            if alpha(j) > H
               alpha(j) = H;
            elseif alpha(j) < L
                   alpha(j) = L;
            end
            if abs(alpha(j)-alphajo)<10^(-5) %the difference is small enough then go to the next loop
                continue;
            end
            alpha(i) = alpha(i) + Ltrain_w(i)*Ltrain_w(j)*(alphajo-alpha(j));
            b1 = b-Ei-Ltrain_w(i)*(alpha(i)-alphaio)*(Dtrain_w(:,i).'*Dtrain_w(:,i))-Ltrain_w(j)*(alpha(j)-alphajo)*(Dtrain_w(:,i).'*Dtrain_w(:,j));
            b2 = b-Ej-Ltrain_w(i)*(alpha(i)-alphaio)*(Dtrain_w(:,j).'*Dtrain_w(:,j))-Ltrain_w(j)*(alpha(j)-alphajo)*(Dtrain_w(:,j).'*Dtrain_w(:,j));
            
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

result = (alpha.*Ltrain_w)*(Dtrain_w.'*Dtest) + b;
result = sign(result)-Ltest;
accuracy(it) = sum(sum(result==0))/Nc/Ntest; %record the accuracy of this classifier

% record the parameters of this classifier
alpha_r(it,:) = alpha;
Ltrain_r(it,:) = Ltrain_w;
Dtrain_r(:,:,it) = Dtrain_w;
b_r(it) = b;

%renew the parameters in AdaBoost
result_0 = (alpha.*Ltrain_w)*(Dtrain_w.'*Dtrain) + b;
epsilon(it) = P*(1-ismember((Ltrain-sign(result_0)),0)).';
if  epsilon(it)>1/2
    disp('error')
end
a(it) = log((1-epsilon)/epsilon)/2;
w = w.*exp(-a(it)*Ltrain.*result_0);
P = w/sum(w);
[bb1,ii1] = sort(-w(1:Ntrain));
[bb2,ii2] = sort(-w((Ntrain+1):Ntrain*Nc));

% choose training data with the highest w in the next iteration
Dtrain_w = Dtrain(:,[ii1(1:Ntrain_w),(ii2(1:Ntrain_w)+Ntrain)]);
Ltrain_w = Ltrain([ii1(1:Ntrain_w),(ii2(1:Ntrain_w)+Ntrain)]);

LL(:,it) = sort([ii1(1:Ntrain_w),(ii2(1:Ntrain_w)+Ntrain)]);


% make sure there still some data points that are not classified correctly
result = 0;
for i = 1:it
    result = result + a(i)*((alpha_r(i,:).*Ltrain_r(i,:))*(Dtrain_r(:,:,i).'*Dtrain) + b_r(i));
end
result = sign(result)-Ltrain;
accuracy_t(it) = sum(sum(result==0))/Nc/Ntrain;
if accuracy_t(it) == 1
    disp(['stop in iteration #',num2str(it)])
    break;
end

end

%%% final classifier %%%%
result = 0;
for i = 1:s
    result = result + a(i)*((alpha_r(i,:).*Ltrain_r(i,:))*(Dtrain_r(:,:,i).'*Dtest) + b_r(i));
end
result = sign(result)-Ltest;
accuracy_r = sum(sum(result==0))/Nc/Ntest; %final accuracy for the testing data

result = 0;
for i = 1:s
    result = result + a(i)*((alpha_r(i,:).*Ltrain_r(i,:))*(Dtrain_r(:,:,i).'*Dtrain) + b_r(i));
end
result = sign(result)-Ltrain;
accuracy0 = sum(sum(result==0))/Nc/Ntrain; %final accuracy for the training data







