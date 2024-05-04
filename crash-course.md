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