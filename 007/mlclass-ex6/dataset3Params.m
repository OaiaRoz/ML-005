function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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

%predictions = svmPredict('gaussianKernel', Xval);
%prediction_error = mean(double(predictions ~= yval))
C_vect = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_vect = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

prediction_error = 1000;

for C_iter = C_vect',
  %fprintf('C_iter: %2.2f \n', C_iter);
  for sigma_iter = sigma_vect',
    %fprintf('  sigma: %.2f \n', sigma_iter);
    
    model = svmTrain(X, y, C_iter, @(x1, x2) gaussianKernel(x1, x2, sigma_iter));
    predictions = svmPredict(model, Xval);
    prediction_error_iter = mean(double(predictions ~= yval));
    %fprintf('  -> prediction_error: %f \n', prediction_error_iter);
    if (prediction_error_iter<prediction_error)
      prediction_error = prediction_error_iter;
      C = C_iter;
      sigma = sigma_iter;
    endif
  end;
  %fprintf('\n');
end;
%fprintf('\n');

% =========================================================================

end
