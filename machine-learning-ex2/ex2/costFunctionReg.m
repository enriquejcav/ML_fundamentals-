function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(X(1,:));
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

g = sigmoid(theta' * X');
g = g' ;    
set=ones(size(theta));
set(1,1)=0;
J = ((1/m)  .*   sum(-1.*(y.*log(g) + (1-y).*log(1-g))))   +   ((lambda/(2*m)).*(sum((theta.^2).*set)));

%set=zeros(size(theta));

%set(1,1) = set(1,1) - (1/m).*sum((y-g).*X(:,1))  ;

%    k=2;
%while (k>=2 && k<=28)    
%    set(k,1) = (set(k,1)) + (lambda/m)*(theta(k,1))  - (1/m).*sum((y-g).*X(:,k))  ;
%    k=k+1;
%end

%grad = set(:,:);
grad(1,1) = grad(1,1) - (1/m).*sum((y-g).*X(:,1))  ;
    k=2;
while (k>=2 && k<=n(1,2))    
    grad(k,1) = (grad(k,1)) + (lambda/m)*(theta(k,1))  - (1/m).*sum((y-g).*X(:,k))  ;
    k=k+1;
end


% =============================================================

end
