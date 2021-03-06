function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
n = size(X, 2); %eu         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


X = [ones(m, 1) X];
h1 = sigmoid(Theta1 * X');
h1=h1';
h1 = [ones(m, 1) h1];
h2 = sigmoid(Theta2 * h1');
h2=h2';
Y=zeros(size(h2));

tamanho=1;
saida=1;
while (saida<=size(h2,2))
    while (tamanho<=length(h2))
        if (y(tamanho,1)==saida)
            Y(tamanho, saida)=1;
        else
            Y(tamanho, saida)=0;
        end
        tamanho=tamanho+1;
    end
    saida=saida+1;
    tamanho=1;
end

set2=ones(size(Theta2));
set2(:,1)=0;
set2=set2';

set1=ones(size(Theta1));
set1(:,1)=0;
set1=set1';

Theta2=Theta2';
Theta1=Theta1';
J = sum(((1/m)  .*   sum(-1.*(Y.*log(h2) + (1-Y).*log(1-h2))))) ...
    + ((lambda/(2*m)).*((sum (sum((Theta2.^2).*set2)) )+(sum (sum((Theta1.^2).*set1)) )));   
%J = sum(J);
%J = J + ((lambda/(2*m)).*((sum (sum((Theta2.^2).*set2)) )+(sum (sum((Theta1.^2).*set1)) )));
Theta2=Theta2';
Theta1=Theta1';


%g = zeros(size(z));
%g1= (1.0 ./ (1.0 + exp(-z)));
%g1=g1';
%g1 = [ones(length(g1), 1) g1];
%g1=g1';
%g = g1.*(1-g1);


delta2 = h2 - Y; 
g1 = h1.*(1-h1);
delta1 = (Theta2'*delta2')'.*g1;
%delta1 = delta1';

delta1 = delta1(:,2:size(Theta1,1)+1);

DELTA1 = delta1'*X;
DELTA2 = delta2'*h1;

D1 = (1/m).*DELTA1;
D2 = (1/m).*DELTA2;


Theta1_grad = D1 ;
Theta2_grad = D2 ;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
