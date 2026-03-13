# Multiple Linear Regression

Multiple Linear Regression is same as a Simple Linear Regression. The only difference is Simple linear regression uses 1 feature to predict output but Multiple Linear regression uses 2 or more features to predict output.

Example of Multiple Linear regression -  
Predicting house price using features `house size` and `number of bedrooms`

Since we have multiple features, each feature will have its own slope (m). y-intercept (b) will be only one.

That is, to predict house price `(y)` using features house size `(x1)` & number of bedrooms `(x2)`.

equation will be `y = (m1 * x1) + (m2 * x2) + b`

After predicting house price (y) for each house size (x1) and its number of bedrooms (x2), we calculate MSE.

```
MSE = ( 1/N ) ∑ $(Actual price - predicted price)^{2}$

Later we calculate gradient of m1 & m2.
```
