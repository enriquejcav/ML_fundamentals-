function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));
%v=length(z);
%k = 1;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
%while k<=v
%    g(k) = 1/(1+exp(-1*(z(k))));
%    k=k+1;
%end

g = 1 ./ (1 + exp(-z));




% =============================================================

end
