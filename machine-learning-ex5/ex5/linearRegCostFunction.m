function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%k=1;

%while (k<=size(y,1))
   
%    J = J + (sum( ((theta(1,1).*X(k,1)+theta(2,1).*X(k,2))-(y(k))).^2))./(2*m);
%    k=k+1;
    
%end

h = theta'*X';
h = h';
set=ones(size(theta));
set(1,:)=0;
J = (  (sum((h-y).^2))   ...
    +   (lambda).*(sum((theta.^2).*set)))...
    ./(2*m);

k=1;
n = size(X, 2);
grad(k,1) = -1*((1/m).*sum((y-h).*X(:,1)))  ;
    k=2;
while (k>=2 && k<=n)    
    grad(k,1) =  -1*((1/m).*sum((y-h).*X(:,k)) - (lambda/m).*theta(k,:))   ;
    k=k+1;
end






% =========================================================================

%grad = grad(:);

end
