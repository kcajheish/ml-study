Model
$$
y = b_0 + \sum_{i=1}^{N}{w_i}{x_i}
$$
- b: bias
- y: inference; label
- w: weight
- x: feature

MSE, mean square error
- loss over entire data set
$$
MSE = {{1 \over{N}} \sum_{x, y}(y-prediction(x))^2 }
$$
- N: number of data in the dataset
- x: feature
- y: label

Iterative learning
- Learn from loss by feeding the data to model, update model parameters, and repeat.
    - loss will gradually converges.
- how to update? -> gradient descent
    - given a set of w and b
    - calculate gradient of loss at w,b
        - gradient: partial derivative with all independent variables
    - multiply negative gradient vector with learning rate
        - loss oscillates when learning rate is large
        - loss vs epoch never converges if learning rate is too low.
    - add value to w, b

Stochastic Gradient descent
- feed the model with a batch of example
    - pro: reduce computation cost to reach next w, b
    - con: result can be noisy
- thus in practice, we use mini-batch
    - usually from 10 to 1000 examples
    - pro: less noisy while keep computation cost low

Epoch
- model is trained over entired dataset
- by having more epoch, mode may be converged.
$$
{1\ epoch} = {N\over{batch\ size}}\ iterations \\
$$

Hyperparameter
- learning rate, batch size, epoc
- we tune them so that loss converges
- rule
    - if loss decreases slowly, increase learning rate
    - if loss doesn't converge(see noise)
        1. decrease learning rate
        2. increase epoch
        3. increase batch size
    - gradully reduce batch size
    - reduce batch size if data can't be fit in memory at once

Overfit
- Complicated model is bad at prediction
    - Keep your model as simple as possible
- Divide our data into test set and training set
    - It doesn't make sense to train our model with infinite data.
    - Note that
        - data should be draw randomly
        - distribution shouldn't change over time
        - we draw data from the same distribution
- We need another split for evaluation set
    - If we update model based on the result of test set alone, this introduces bias.

Correlation matrix shows which feature has most impact on prediction.
    - training_df.corr()
    - Correlation value shows predicative power of a feature. The larger the absolute value, the more impacts it has on prediction.
        - 1: perfect positive
        - -1: perfect negative
        - 0: no correlation

Split your data
- trainning set
    - examples that are used to train the model
- testing set
    - examples that test your model
- Make sure you draw randomly for your testing set.
- Make sure testing set is large enough.
- Don't train the model with your testing set.
- Experiment
    - High learning rate causes test loss is higher than training loss
    - Batch size doesn't influence test loss that much
    - Smaller split leads to lots of jump in the loss curve.

Validation set
- validate the trained model with validation set; tweak the model if validation loss deviated from training loss too much
- It works since test set isn't exposed when we tweak the model.

Representation
- Feature engineering
    - feature vector
        - Extract feature from multiple data sources.
- hot encoding
    - assign a number to the string; create a feature vector based on those numbers
        - e.g. "Main stree" has code of 1. -> [0, 0, ..., 1, ...]
    - one hot encoding
        - one 1 in the vector
    - multi-hot encoding
        - multiple 1 in the vector
    - use sparse vector when vector size reaches 1 million
- To have good feature
    - Avoid rarely used value
        - e.g. my_device_id is a bad feature
    - obvious meaning
        - e.g. use year over second to represent age
    - no magic number
        - e.g. house is never on the market
            - bad: watch_time = -1
            - good: watch_time_is_defined = false
    - feature shouldn't change over time
    - get rid of outlier
- scale
    - turn feature value into [-1, 1]
    - pro
        - model converges quickly
        - avoid NaN
            - value exceeds floating point precision
        - model learns each feature with equal efforts
- handle outlier
    - tail in the histogram could be outlier
    - possible solution
        - take log
        - cap a threshold
        - note that both didn't ignore the outlier but rather let model learn too much from the outlier.
- bin
    - feature doesn't scale linearly with label; instead their relationship is discrete.
        - e.g. latitude vs number of house
    - Thus, a vector of bins can represent this feature
        - e.g. [1, 0] north pole has one house but south pole doesn't
- scrub
    - visualize data with histogram
    - remove bad example
        - mislabel
        - bad feature
        - duplication
        - empty value
- feature crosses
    - use linear model to predict non-linear behavior
        - Set $x_3 = x_2*x_1$
        - Then we have model
        $$
            y = b + w_1x_1 + w_2x_2 + w_3x_3
        $$
    - why not non-linear model?
        - it's efficient to train large dataset with linear model
    - what do feature crosses look like?
    ```
    binned_latitude(lat) = [
        0  < lat <= 10
        10 < lat <= 20
        20 < lat <= 30
    ]

    binned_longitude(lon) = [
        0  < lon <= 15
        15 < lon <= 30
    ]

    binned_latitude_X_longitude(lat, lon) = [
        0  < lat <= 10 AND 0  < lon <= 15
        0  < lat <= 10 AND 15 < lon <= 30
        10 < lat <= 20 AND 0  < lon <= 15
        10 < lat <= 20 AND 15 < lon <= 30
        20 < lat <= 30 AND 0  < lon <= 15
        20 < lat <= 30 AND 15 < lon <= 30
    ]
    ```


Generalization curve shows loss(iteration) for both training and validation set.
- You can observe overfit if training loss decreases but validation loss increases. This is because the model is complex and we try to fit outlier.
- Thus, loss alone can not describe the quality of prediction. We have to account for model complexity.

Regularization describes complexity of the model. The parameters of the complexity can be:
1. weight
2. number of feature with non-zero weight

$L_2$ regularization describes model complexity with sum of $weight^2$
- The larger the absolute value, the more complex the model is.
- To minimize it,
    - weight is zero
    - the mean of weights is zero

Lambda can be included in regularization term.
- $Loss + \lambda\ complexity(model)$
    - The larger the lambda is, the simpler the model is
        - For a simple model, weight histogram has bell-shaped or Gaussian.
        - for a complex model, weight historgram has flat distribution
- note that if lambda is too large, the model may underfit the training data

We may predict probability that an event happens. Thus, logistic regression is used.
- To ensure output of model is between 0 and 1, use sigmoid function
    - $y^{'} = {1\ \over 1 + e^{-z}}$
        - y: probability
        - $z = b + w_0 x_0 + w_1 x_1+...+ w_n x_n$
        - log odd
            - $z = {\log{y\over(1-y)}}$

loss of logistic regression
- tbc

Classification
- a classification threshold turns a probability into binary value
    - e.g. Let the threshold be 0.6, then 0.9 predicted by logistic regression means a spam.
- confusion matrix
    - true positive
        - e.g. wolf is coming and shepherd alerts
    - false positive
        - e.g. wolf is not coming and shepherd alerts
    - true negative
        - e.g. wolf is comming but shepherd doesn't alert
    - false negative
        - e.g. wold is not coming but shepherd alerts
    - note
        - positive/negative
            - whether an event occurs or not
        - true/false
            - whether model predicts outcome correctly or not
- Accuracy is used to evaluate classification model
    - percentage of correct predictions
        $$
        \begin{align}
            &= {{number\ of\ correct\ predictions}\over{number\ of\ total\ predictions}} \\
            &={{TP + TN}\over{TP+FP+TN+FN}}
        \end{align}
        $$
    - con
        - doesn't work for class-imbalanced dataset
            - e.g. 99 out of 100 examples are malignant. A model that always predict malignant will achieve 99% accuracy.
- Precision
    - portion of positive that is correct
    $$
        Precision = {TP \over {TP + FP}}
    $$
- Recall
    - portion of actual positive that is predicted
    $$
        Recall = {TP \over {TP + FN}}
    $$
- Often you can't have both high precision and recall
    1. You set high threshold. Classification model is not likely to report spam that isn't but a lot of spams are treated as not spam. Thus, you have high precision but low recall.
    2. You set low threshold. Then you have low precision but high recall.

