% #6 MDA: implement MDA followed by the training/application of the Bayes’ Classifier
%    Identify the subject label
clear; clc; close all

s = input('Please input a number: #1 data, #2 pose, #3 illumination: ');
if ismember(s,[1,2,3])~=1
    disp('error.')
    return
end

if s==1
    load('./DATA/data.mat');
    face_r = reshape(face,24*21,600); %reshape the face
    Ntrain = 2;
    Ntest = 1;
    Dtrain = face_r(:,sort([3*(1:200)-2,3*(1:200)-1])); %use the 1st and 2nd pictures for training
    Dtest = face_r(:,3*(1:200)); %use the 3rd pictures for testing
    Nc = 200; %number of classes
    Nd = 24*21; %dimension
    
    
    
elseif s==2
    load('./DATA/pose.mat');
    pose_r = reshape(pose,48*40,13,68);
    Ntrain = 6; %the number of pose used for training in each class
    Ntest = 13-Ntrain; %the number of pose for testing in each class
    Dtrain = zeros(48*40,Ntrain*68);
    Dtest = zeros(48*40,Ntest*68);
    for i = 1:68
        for j = 1:Ntrain
            Dtrain(:,Ntrain*(i-1)+j) = pose_r(:,j,i);
        end
        for k = 1:Ntest
            Dtest(:,Ntest*(i-1)+k) = pose_r(:,k+Ntrain,i);
        end
    end
    Nc = 68; %number of classes
    Nd = 48*40; %dimension
           
    
    
    
else
    load('./DATA/illumination.mat');
    Ntrain = 10; %the number of illum used for training in each class
    Ntest = 21-Ntrain; %the number of illum for testing in each class
    Dtrain = zeros(1920,Ntrain*68);
    Dtest = zeros(1920,Ntest*68);
    for i = 1:68
        for j = 1:Ntrain
            Dtrain(:,Ntrain*(i-1)+j) = illum(:,j,i);
        end
        for k = 1:Ntest
            Dtest(:,Ntest*(i-1)+k) = illum(:,k+Ntrain,i);
        end
    end
    Nc = 68; %number of classes
    Nd = 1920; %dimension
    
      
end

m = floor(Nd/2);

mui = zeros(Nd,Nc);
for i = 1:Nc
    mui(:,i) = sum(Dtrain(:,(i-1)*Ntrain+(1:Ntrain)),2)/Ntrain;
end

Sigmai = zeros(Nd,Nd,Nc);
for i = 1:Nc
    for j = 1:Ntrain
        Sigmai(:,:,i) = Sigmai(:,:,i) + (Dtrain(:,(i-1)*Ntrain+j)-mui(:,i))*(Dtrain(:,(i-1)*Ntrain+j)-mui(:,i)).'/Ntrain;
    end
end

Sigmaw = sum(Sigmai,3)/Nc;
mu0 = sum(mui,2)/Nc;

Sigmab = zeros(Nd,Nd);
for i = 1:Nc
    Sigmab = Sigmab + (mui(:,i)-mu0)*(mui(:,i)-mu0).'/Nc;
end

%%%%%%%%%% Method 1: project Dtrain to a space from Sigmaw to avoid the case that matrix is not full rank %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [Uw,Dw] = eigs(Sigmaw,rank(Sigmaw));
% Dtrain = Uw.'*Dtrain;
% Dtest = Uw.'*Dtest;
% Nd = rank(Dw);
% m = floor(Nd/2);
% 
% mui = zeros(Nd,Nc);
% for i = 1:Nc
%     mui(:,i) = sum(Dtrain(:,(i-1)*Ntrain+(1:Ntrain)),2)/Ntrain;
% end
% 
% Sigmai = zeros(Nd,Nd,Nc);
% for i = 1:Nc
%     for j = 1:Ntrain
%         Sigmai(:,:,i) = Sigmai(:,:,i) + (Dtrain(:,(i-1)*Ntrain+j)-mui(:,i))*(Dtrain(:,(i-1)*Ntrain+j)-mui(:,i)).'/Ntrain;
%     end
% end
% 
% Sigmaw = sum(Sigmai,3)/Nc;
% mu0 = sum(mui,2)/Nc;
% 
% Sigmab = zeros(Nd,Nd);
% for i = 1:Nc
%     Sigmab = Sigmab + (mui(:,i)-mu0)*(mui(:,i)-mu0).'/Nc;
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%% Method 2: add a noise %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Sigmaw = Sigmaw + eye(Nd);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


[U,D] = eigs(Sigmab,Sigmaw,m);

Dtrain = U.'*Dtrain;
Dtest = U.'*Dtest;



mu = zeros(m,Nc);
for i = 1:Ntrain
    mu = mu+Dtrain(:,((1:Nc)-1)*Ntrain+i)/Ntrain;
end

Sigma = zeros(m,m,Nc);
Sigma_inv = zeros(m,m,Nc);
for j = 1:Nc
    for i = 1:Ntrain
        Sigma(:,:,j) = Sigma(:,:,j) + (Dtrain(:,(j-1)*Ntrain+i)-mu(:,j))*(Dtrain(:,(j-1)*Ntrain+i)-mu(:,j)).'/Ntrain;

    end
        Sigma(:,:,j) = Sigma(:,:,j)+eye(m); %add a small function here to make the inverse always exists
        Sigma_inv(:,:,j) = inv(Sigma(:,:,j));
        if det(Sigma(:,:,j))==0
            disp(['error in ', num2str(j)])
        end
end


omega = zeros(m,Nc);
omega0 = zeros(1,Nc);
for i = 1:Nc
    omega(:,i) = Sigma_inv(:,:,i)*mu(:,i);
    omega0(i) = -mu(:,i).'*Sigma_inv(:,:,i)*mu(:,i)/2-log(det(Sigma(:,:,i)))/2;
end
W = -Sigma_inv/2;

%test
result = zeros(1,Ntest*Nc);
g = zeros(1,Nc);
for i = 1:Nc*Ntest
    for j = 1:Nc
        g(j) = Dtest(:,i).'*W(:,:,j)*Dtest(:,i)+omega(:,j).'*Dtest(:,i)+omega0(:,j);
    end
    [a,result(i)] = max(g);
    result(i) = result(i)-ceil(i/Ntest);
end
accuracy = sum(sum(result==0))/Nc/Ntest;




