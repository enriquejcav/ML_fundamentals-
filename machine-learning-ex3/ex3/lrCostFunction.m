function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(X, 2);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
g = sigmoid(theta' * X');
g = g';
set=ones(size(theta));
set(1,1)=0;
J = ((1/m)  .*   sum(-1.*(y.*log(g) + (1-y).*log(1-g))))   +   ((lambda/(2*m)).*(sum((theta.^2).*set)));

%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

 

grad(1,1) = -1*(grad(1,1) - (1/m).*sum((g-y).*X(:,1)))  ;
    k=2;
while (k>=2 && k<=n)    
    grad(k,1) = (grad(k,1)) + (lambda/m)*(theta(k,1))  - (1/m).*sum((y-g).*X(:,k))  ;
    k=k+1;
end







% =============================================================
%grad = set(:,:);
%grad = grad(:);

end
