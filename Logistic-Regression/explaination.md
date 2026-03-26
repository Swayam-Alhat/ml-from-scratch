# Explanation of Logistic Regression

Logistic Regression is ML algorithm used for classification. Unlike linear regression which predicts continuous values it predicts the probability that an input belongs to a specific class.

Example, to predict if email is spam or not spam.

To know more about logistic regression, Read article from IBM on [Logistic Regression](https://www.ibm.com/think/topics/logistic-regression)

_Since, probability lies between 0 and 1. we need to calculate value of y (target variable) which lies within 0 and 1._

Thus, we use **sigmoid function** which outputs value between 0 and 1. Sigmoid function is also known as Logistic function.

## Derivation of Sigmoid function

We cannot use linear function  
 y = $b_{0}$ + $b_{1}$ . $x_{1}$  
because this outputs value of y (target variable) within (-∞ , ∞). But Logistic regression predicts probability of outcome and we know probability lies between 0 and 1. So we work with _odds_.

- _Probability_ : Probability measures the chance of an event occurring out of all possible outcomes.

![probability-figure](./figures/probability-fig.png)

- _Odds_ : Odds compare the chance of an event occurring to the chance of it not occurring.

![odds-figure](./figures/odds-figure.png)

> In logistic Regression, we work with odds since we have to predict if outcome is **favourable** or **not favourable**.

So, if we take odds of input x

$$
\text{odds(x)} =\frac{p(x)}{1 - p(x)}
$$

Odds ranges from 0 to ∞.

- An odds value greater than 1 indicates a favorable outcome. means p > 0.5, i.e. the event is more likely than not
- less than 1 indicates an unfavorable outcome. Means the event is less likely than not.
- equal to 1 means the event is just as likely to occur as not.

_To address this imbalance, we take the logarithm of the odds, which transforms range of odds from (0 ,∞) to the real number line (−∞, ∞). This is known as the **log-odds**, or logit, and is the foundation of the logistic regression model_

Since, Log-odds outputs value within (−∞, ∞), we can say Linear function y = $b_{0}$ + $b_{1}$ $x_{1}$ is equal to log-odds.

> Because $b_{0}$ + $b_{1}$ . $x_{1}$ and log-odds outputs value which lies within (−∞, ∞).

$`\text{log} \text{(}\frac{p(x)}{1 - p(x)} \text{)}`$ = $b_{0}$ + $b_{1}$. $x_{1}$

We can then exponentiate both sides to get back to odds:

$`\frac{p(x)}{1 - p(x)}`$ = $e^{b_0 + b_1 . x_1}$

Solving for `p(x)`we get the _Sigmoid function_, which helps ensure the predicted value stays between 0 and 1:

p(x) = $`\frac{e^{b_0 + b_1 . x_1}}{1 + e^{b_0 + b_1 . x_1}}`$

> [!NOTE]  
> The above form is equivalent to the more common compact form of Sigmoid function:
> $$p(x) = \frac{1}{1 + e^{-(b_0 + b_1 x_1)}}$$
> Multiply numerator and denominator by $`e^{-(b_0 + b_1 . x_1)}`$ and solve it. We will get commonly used Sigmoid function form

This transformation allows logistic regression to output valid probabilities, even though we’re modeling them using a linear function underneath.

## Odds Ratio

_The odds ratio tells us how the odds change when the input variable x1 increases by one unit._

Example, odds of $`x_1`$ is

$`odds(x_1)`$ = $`e^{b_0 + b_1 . x_1}`$

If we increase $`x_1`$ by 1 unit, we get

$`odds(x_1 + 1)`$ = $`e^{b_0 + b_1 . ( x_1 + 1 )}`$

$`odds(x_1 + 1)`$ = $`e^{b_0 + b_1 x_1} . e^{b_1}`$

This means that for every one-unit increase in $`x_1`$, the odds are multiplied by $`e^{b_1}`$ . This multiplier is the _odds ratio_.

- If $`b_1`$ > 0, then $`e^{b_1}`$ > 1, odds increase (event becomes more likely)
- If $`b_1`$ < 0, then $`e^{b_1}`$ < 1, odds decrease (event becomes less likely)
- If $`b_1`$ = 0, then $`e^{b_1}`$ = 1, odds unchanged (input has no effect)

## How Logistic Regression algorithm learns

Since we know logistic regression is used for classification, it calculates probability of whether given input `x` belongs to class 1 or not.

_So we need a function which outputs value between 0 and 1.  
(since probability is between 0 and 1_)

linear function `y =  mx + b` outputs value between −∞ to ∞. So we need a function which transforms it into value which is between 0 and 1

**Sigmoid does it**

Standard formula of sigmoid -

$`σ(z) = \frac{1}{1 + e^{-z}}`$

where, $`z`$ is linear prediction. Meaning predicted value using `z =  mx + b`

So, basically, $`z`$ is output of `mx + b`

So we can say,  
$`z`$ = $`mx + b`$

where `m` is weight and `b` is bias`

let $`z`$ be the predicted value using linear function $`z`$ = $`mx + b`$

So, initially, It does the same as linear regression does.

Calculates $`z`$ using linear function $`z`$ = $`mx + b`$

Then apply Sigmoid function on it

σ ($`z`$) = $`\frac{1}{1 + e^{-z}}`$

After calculating, we get value of σ($`z`$) which is between 0 and 1. **This is estimated probability**.

Let $`y_p`$ be the estimated probability

So, σ($`z`$) = $`y_p`$

Now, we compare $`y`$ (actual value) and $`y_p`$ (predicted value)

- If $`y`$ is 1 and $`y_p`$ is close to 1, we say our prediction is **correct**.

- If $`y`$ is 0 and $`y_p`$ is close to 0, we say our prediction is **correct**.
- If $`y`$ is 1 and $`y_p`$ is close to 0, we say our prediction is **incorrect**.
- If $`y`$ is 0 and $`y_p`$ is close to 1, we say our prediction is **incorrect**.

_Correct prediction means current values of `m` and `b` are optimal._

_Incorrect prediction means current values of `m` and `b` are not optimal_

**Now, $`y_p`$ is predicted value for current values of m and b (weights and bias)**

But, to find the optimal values of m and b, we need a function/equation that tells the likelihood that current weights are being optimal
