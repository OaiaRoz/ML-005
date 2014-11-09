function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
% h(X(iter,:)) = sigmoid(X(iter,:)*theta);
%
cost = 0;
for iter = 1:m
  cost = cost + y(iter)*log(sigmoid(X(iter,:)*theta)) + (1 - y(iter))*log(1 - sigmoid(X(iter,:)*theta));
end
J = -cost/m;

for iter = 1:m
  grad = grad + (sigmoid(X(iter,:)*theta) - y(iter)) * X(iter, :)';
end

grad = (1/m) * grad;


% =============================================================

end