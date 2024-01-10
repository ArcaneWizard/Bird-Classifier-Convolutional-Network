# Bird-Classifier-Convolutional-Network

The dataset used contained 81950 images and 510 bird labels. 30% of training dataset was used for training, while a separate 2550 images were used for validation, and another 2550 for testing.

## Model
The loss function used cross-entropy loss, which is common for multi-class classification problem
It strongly penalizes assigning a low probability to the correct class, and on top of that, is is differentiable. This makes it a good pick for this project.

The functions for training and prediction passes were taken from UW's Machine Learning course's example PyTorch notebook (up till lines 145). The rest of the code till line 320 was self-implemented, including a 7-layer convolutional network model, and employing hyperparameter tuning to select the top 3 configurations before further training the top configurations for more epochs. The most unique hyperparameter was varying the number of output channels in the hidden convolutional layers.

The final model included hyperparameter tuning to determine the number of output channels for each convolution, learning rate, and weight decay. Other choices like RELU vs ELU, SGD vs Adam, or choosing the right batch size were tested beforehand, and analyzed afterwards. An analysis is linked below:

[Analysis](https://raghavnarula77.wixsite.com/bird-classifier---co)
