function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
##C = 1;
##sigma = 0.3;

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

%Setting the range of C and sigma values
C_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_list = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

%Creating the matrix of error for each C and sigma set
error_val = zeros(length(C_list),length(sigma_list));

%Looping to get all error for each C and sigma set upon cross-validation data
for i = 1:length(C_list)
  for j = 1:length(sigma_list)
   model = svmTrain(X, y, C_list(i), @(x1, x2) gaussianKernel(x1, x2, sigma_list(j)));
   error_val(i,j) = mean(double(svmPredict(model, Xval) ~= yval)); 
  endfor
endfor

%Taking the C and sigma set with the lowest cross training error
[Min, indices] = min(error_val(:));
C = C_list(mod(indices - 1,length(C_list))+1);
sigma = sigma_list(((indices - mod(indices-1,length(C_list)) - 1)/length(sigma_list))+1);


% =========================================================================

end
