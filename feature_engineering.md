why feature engineering?
- We need good data so that model learns the right thing and makes good prediction.

construct dataset
- step
    - collect raw data
    - feature and label
    - sampling method
    - split data
- Data collection pipline
    - starts with few features to establishes baseline

# Transform data

We need good data so that model can find pattern in those data easily.

Before we train model, we have to construct and transform dataset
- construct
    - split data
    - sample
    - collect
    - make feature/label
- transform
    - clean
    - feature engineering
        - def: create feature from multiple sources
why transform data?
- matrix multiplication has fix input and operate on numeric. thus we need:
    1. convert string to numeric value
    2. resize input feature
- other purpose:
    1. express nonlinearity in linear model
        - b/c nonlinear model like neural network takes a lot of computation to train
    2. normalize the data
        - b/c gradient descent won't bounce around and converge quickly
    3. lower casing
        - b/c turn letters into upper case shouldn't affect prediction

transform data before training
- pro
    - compute once
    - easy to lookup entire dataset
- con
    - skew data for online/offline serving
        - e.g. you can't find the vocabulary, which was built in offline for online feature
    - slow iteration
        - have to change every data at once rather than on demand

transform data within training
- pro
    - transformation is the same for online and offline
    - Source files can be reused even if you need a different transformation
- con
    - model latency increases

Before we transform data, explore the data a bit
1. visualize it
2. check statistic

ways to transform data
1. normalize
2. bucketing

we normalize the data b/c
1. gradient descent converges quickly and doesn't bounce around
2. feature with wide range makes model return NAN in gradient descent

There are many types of normalization which compress the range of the dataset
- scale to a range
    - eq
    $$
        x^{'}\ = {x-x_{min} \over x_{max}-x_{min}}
    $$
    - when data are distributed evenly
        - e.g. age
        - not good for income
- log scaling
    - eq
    $$
        x^{'}\ = \log(x)
    $$
    - when data are distributed according to power law
- feature clipping
    - eq
    $$
        \begin{align}
            x = min(x, upper_bound) \\
            x = max(x, lower_bound)
        \end{align}
    $$
    - when many outliers are far from majority of the data
- z scale
    - turn mean to zero and make deviation 1
    - eq
        $$
            x^{'} = {x-\mu \over \sigma}\\
        $$
        - where
            - $\sigma$: std, standard deviation
            - $\mu$: mean

    - when data has few outlier and follow normal distribution

Bucketing
- given a set of threshold, convert numeric value to category data
    - fix space bucket
        - divide value into equal range
    - quantile bucket
        - each bucket has the same number of point
        - some buckets may have wider span

Category feature has discrete value. We usually assign a category to a numeric based on mapping

One hot encoding represent category feature with binary vector so that model doesn't spend time on learning order and numeric relationship from the category
- e.g. postal code 240 have no meaningful relationship with code 241

out of volcabulary
- a bucket that holds category that are not on the mapping

use hash to create vocabulary
- pro: don't have to select vocabulary for large distribution
- con: hash collision, two category may have nothing in common but are put into the same category


hybrid approach(vocabulary + hashing)
- build up vocabulary for common data
- use hashing for data that keep changing
