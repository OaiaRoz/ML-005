function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%h = X*theta;
%squareErr = h - y;          % errors -> difference between estimated value and real value from training set
%squareErr = squareErr.^2;   % square the errors
squareErr = (X*theta - y).^2;   % square the errors

sumOfSquaredErrors = sum(squareErr);  % summ the squared errors

J = sumOfSquaredErrors/(2*m);  % result



% =========================================================================

end
