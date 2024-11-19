# Enhancements in Actuarial Non-Life Pricing Models: Improved Results and Faster Training
I achieved better results and significantly faster training compared to my previous publication:
https://doi.org/10.1007/s13385-024-00388-2

The main ideas to follow here are:
<!-- <small><i> -->
* Using the Nadam optimizer instead of Wadam for FTT and LocalGLMftt.
* Using Keras 3 with PyTorch/TensorFlow/JAX as the backend instead of Keras 2 with TensorFlow.
* Using the Keras MHSA layer instead of our own (the parameter count and performance are similar, but the fit is faster on Colab).
* Using the .fit() function instead of the custom training loop.
* Using JAX as the backend instead of TensorFlow.
* Setting the GLM weights as trainable in the case of LocalGLMftt.
* Using an L4 instead of a T4 GPU.
<!-- </i></small> -->

## Comparison of previous FTT with new versions using different backends: 
Average of model result over 15 fits with different seeds

![alt text](img\improvement_ftt_1.png)

Average results over 3 ensembles consisting of each 5 rebalanced models

![alt text](img\improvement_ftt_2.png)

* The number of trainable parameters is slightly less than in my previous paper because I didn't include an OOV token ("Out-Of-Vocabulary") in the Cat_Embedding Layer in the new implementation. We had 2 categorical features and the embedding dimension (emb_dim) is 32. Therefore, the new number of trainable parameters should be 64 less than the old one, which is 27,069 instead of 27,133 in the paper.
* Compared to the TensorFlow implementation in my previous paper, the model fits twice as fast in Keras 3 with Torch and more than ten times faster when using Keras 3 with TensorFlow or JAX. While all new models perform better, the performance of the ensemble models is significantly better in the case of JAX.


## Comparison of previous LocalGLMftt with new versions using different backends: 
Average of model result over 15 fits with different seeds

![alt text](img\improvement_localGLMftt_1.png)

Average results over 3 ensembles consisting of each 5 rebalanced models

![alt text](img\improvement_localGLMftt_2.png)

* The number of trainable parameters is slightly less than in my previous paper because:
    * I didn't include an OOV token ("Out-Of-Vocabulary") in the Cat_Embedding Layer in the new implementation. We had 2 categorical features and the embedding dimension (emb_dim) is 32. Therefore, the new number of trainable parameters should be 64 less than the old one.
    * However, due to performance reasons, we chose to make the initial GLM weights trainable, which added 9 more parameters: 7 for the numerical features, 2 for the GLM intercept, and none for the categorical features, as they were already trainable in the previous model.
    * As a result, we end up with 55 fewer weights than before (64 - 9), which is 27,375 instead of 27,430 in the paper.

* Compared to the TensorFlow implementation in my previous paper, the model fits similarly fast in Keras 3 with Torch and more than ten times faster when using Keras 3 with TensorFlow or JAX. While all new models perform better, the performance of the ensemble models is significantly better in the case of Torch and JAX.
