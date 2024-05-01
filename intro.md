Machine learning uses mathematical model to draw relation between the data and then makes preductions.

Supervised learning
- Model makes prediction after seeing a alot of data with correct answers.

Regression
- Model predicts a numeric value

Classification
- Model preducts whether an item belongs to a category or not.
- Binary
    - true or false
- Multi class
    - output a value defined in the class

Unsupervised learning
- Model finds the pattern in the given data without the correct answer.
- Clustering groups dataset with the rules it defines.


Reinforcement learning
- Model makes predictions based on penalties and rewards.
- Policy defines the best strategy to have most rewards.

Generative AI
- Model is trained on given input then generates output based on the pattern it learns.
- Generally, model is initially trained with unsupervised learning and then supervised and reinforcement learning


Feature
- A row of data in the dataset. It can used to train a model and preduct the label.

Label
- The outcome or answer of the model prediction.

Example
- It's rows of data that can include feature and label.
- An unlabeled example doesn't have label, but we can label them after model is trained with dataset.

A dataset should have large diversity and size so that model is well trained and makes good prediction.

To train the model, features are fed to model so that model gradually learns how to predict label.

The quality of predictions is measured by loss
- Loss is the difference between predicted label and actual label.
    - If loss is large, we update the model by adjusting predicator.
        - e.g. remove few columns

Predictions are called Inference.
