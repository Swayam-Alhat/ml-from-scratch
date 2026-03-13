# Multiple Linear Regression

> [!NOTE]
> Before implementing algorithm, I have implemented data preprocessing and feature scaling

Multiple Linear Regression is same as a Simple Linear Regression. The only difference is Simple linear regression uses 1 feature to predict output but Multiple Linear regression uses 2 or more features to predict output.

Example of Multiple Linear regression -  
Predicting house price using features `house size` and `number of bedrooms`

Since we have multiple features, each feature will have its own slope (m). y-intercept (b) will be only one.

That is, to predict house price `(y)` using features house size `(x1)` & number of bedrooms `(x2)`.

equation will be `y = (m1 * x1) + (m2 * x2) + b`

After predicting house price (y) for each house size (x1) and its number of bedrooms (x2), we calculate MSE.

MSE = ( 1/N ) ∑ $(Actual - predicted)^{2}$

Later we calculate gradient of m1, m2 and b

```
gradient of m1 = (-2/N) ∑ (actual - predicted) * x1
gradient of m2 = (-2/N) ∑ (actual - predicted) * x2
gradient of b  = (-2/N) ∑ (actual - predicted)
```

After calculating gradient of m1,m2 and b, update values of m1, m2 and b.

Update formula -

```
m1 = m1 - learning_rate * gradient of m1
m2 = m2 - learning_rate * gradient of m2
b  = b  - learning_rate * gradient of b
```

Repeat

`predict y` -> `Calculate MSE` -> `calculate gradients` -> `update weights and bias`

Until MSE stops changing or changes very slightly
