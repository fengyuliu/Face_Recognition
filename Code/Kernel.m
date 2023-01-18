function result = Kernel(a,b,x,s)
n1 = size(a,2);
n2 = size(b,2);
result = zeros(n1,n2);
if s==1
    for i = 1:n1
        for j = 1:n2
            result(i,j) = exp(-norm(a(:,i)-b(:,j))/(2*x^2));
        end
    end
elseif s==2
    for i = 1:n1
        for j = 1:n2
            result(i,j) = (a(:,i).'*b(:,j)+1)^x;
        end
    end
end