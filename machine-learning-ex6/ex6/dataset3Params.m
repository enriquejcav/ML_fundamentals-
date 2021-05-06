function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C = 1;
%sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
prob=zeros;
Ctest=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma2test=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30].^(1/2);
i=1;
k=1;
cont=1;
while (cont<=length(Ctest)^2)
    model= svmTrain(X, y, Ctest(k), @(x1, x2) gaussianKernel(x1, x2, sigma2test(i)));
    predictions = svmPredict(model, Xval);
    prob(k,i) = mean(double(predictions ~= yval));
    if((mod(k,length(Ctest)))==0)
        i=i+1;
        k=0;
    end
    k=k+1;
    cont=cont+1;
end
[M I] = min(prob,[],1);
[MM II] = min(M,[],2);
C = Ctest (I(II));
sigma = sigma2test(II);

% =========================================================================

end
