# Data Drift Analysis using Deep Learning and Novelty Detection 
### A model agnostic way to detect data drift in production data after model deployment

#### For details - Read the full article

## Installation 
- Python 3.7+
- Keras/Tensorflow
- Sklearn

### Code explanation
In this example code, we first load the training data and define the autoencoder model using Keras. We then train the autoencoder model on the training data.

Next, we extract the encoded features using the encoder model and train the novelty detection model using the LOF algorithm from Sklearn. We use the encoded features to train the novelty detection model because they represent the important features of the training data.

To simulate data drift, we load new data and predict whether the samples belong to the training set or not using the novelty detection model. If we detect data drift, we can update the model as necessary. If no data drift is detected, we can continue to use the current model.

This example code is just a simplified demonstration of how to implement the model-agnostic data drift detection strategy using Keras and Sklearn. Depending on your specific use case, you may need to modify the code accordingly.
