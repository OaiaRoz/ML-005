function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
% h(X(iter,:)) = sigmoid(X(iter,:)*theta);
%
cost = 0;
for iter = 1:m
  cost = cost + y(iter)*log(sigmoid(X(iter,:)*theta)) + (1 - y(iter))*log(1 - sigmoid(X(iter,:)*theta));
end
%J = -cost/m + (lambda/(2*m))*sum(theta(2:size(theta,2)) .* theta(2:size(theta,2)));

interm_sum = 0;
for iter = 2:size(X,2)
  interm_sum = interm_sum + theta(iter)*theta(iter);
end

J = -cost/m + (lambda/(2*m))*interm_sum;

reg_theta = zeros(size(theta));
reg_theta = reg_theta + theta;
reg_theta(1) = 0;

for iter = 1:m
  grad = grad + (sigmoid(X(iter,:)*theta) - y(iter)) * X(iter, :)';
end

grad = (1/m) * grad  + (lambda/m) * reg_theta;

%for iter = 2:size(X,2)
%  grad(iter) = grad(iter) + (lambda/m)*theta(iter);
%end



% =============================================================

end
