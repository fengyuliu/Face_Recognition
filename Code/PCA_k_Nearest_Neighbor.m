% #5 PCA: Implement Principal Component Analysis (PCA) and apply it to the data before feeding it into 
%         k-NN Rule.
%    Identify the subject label
clear; clc; close all;

s = input('Please input a number: #1 DATA, #2 pose, #3 illumination: ');
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
Dmu = mean(Dtrain,2);
Dtrain = Dtrain-Dmu*ones(1,Ntrain*Nc);
Cov = Dtrain*Dtrain.'/Nc/Ntrain;
[U,S,V] = svds(Cov,m);

Dtrain = U.'*Dtrain;
Dtest = U.'*(Dtest-Dmu*ones(1,Ntest*Nc));

k = 1; %number of the nearest nerghbors we considered
distance = zeros(Ntrain*Nc,Ntest*Nc);
for i = 1:Ntest*Nc
    for j = 1:Ntrain*Nc
        distance(j,i) = (Dtest(:,i)-Dtrain(:,j)).'*(Dtest(:,i)-Dtrain(:,j));
    end
end

result_knn = zeros(Ntest*Nc,k);
result = zeros(Ntest*Nc,1);
for i = 1:Ntest*Nc
    [a,b] = sort(distance(:,i));
    result_knn(i,:) = ceil(b(1:k)/Ntrain);
    result(i) = mode(result_knn(i,:))-ceil(i/Ntest); %When the number of repetitions is the same, the smallest value will be returned
end
accuracy = sum(sum(result==0))/Nc/Ntest;

