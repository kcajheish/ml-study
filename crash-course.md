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
    - Extract feature from multiple data sources.
- To have good feature
    - one hot encoding
        - assign a number to the string; create a feature vector based on those numbers
            - e.g. "Main stree" has code of 1.
    - non zero feature should appear sometimes
        - e.g. my_device_id is a bad feature
    - obvious meaning
        - e.g. use year over second to represent age
    - no magic number
        - e.g. house is never on the market
            - bad: watch_time = -1
            - good: watch_time_is_defined = false
    - feature shouldn't change over time
    - get rid of outlier
    - divide your data into mulitple bin over a feature before you train
- to know your data
    - debug
    - visualize
    - monitor
