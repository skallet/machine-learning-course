function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.01;
sigma = 0.01;
best = [10;C;sigma];

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

T = [0.01;0.03;0.1;0.3;1;3;10;30];

for c = 1:rows(T)
  C = T(c);
  for s = 1:rows(T)
    sigma = T(s);
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    err = mean(double(predictions ~= yval))
    %printf("Calc C = %.4f; sigma = %.4f; Error = %.6f;\n", C, sigma, err);
    if (best(1) > err)
      best = [err;C;sigma];
    endif
    
    sigma = sigma * 2;
  endfor
  C = C * 2;
endfor

C = best(2);
sigma = best(3);
printf("Best C = %.4f; sigma = %.4f;\n", C, sigma);


% =========================================================================

end
