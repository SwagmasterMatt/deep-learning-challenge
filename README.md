# deep-learning-challenge

## Alphabet Soup Charity Deep Learning Model Step 1

Purpose - The goal of this analysis was to predict wether or not money was adequately spent from the charity.

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively

IS_SUCCESSFUL is our predictor, y


### Initial CSV
![image](https://github.com/SwagmasterMatt/deep-learning-challenge/assets/133460903/26c3a8be-22ba-43d6-9af8-edef78f386bf)

### Data Pre Processing
![image](https://github.com/SwagmasterMatt/deep-learning-challenge/assets/133460903/8848f711-f688-4cd7-b9ba-a583e53b9098)
--------------------------------------------------------------------------
![image](https://github.com/SwagmasterMatt/deep-learning-challenge/assets/133460903/beb38a38-6b5a-4784-a1bd-c21ce946af72)
--------------------------------------------------------------------------
![image](https://github.com/SwagmasterMatt/deep-learning-challenge/assets/133460903/7b5372e2-e4f9-43ba-b6c8-2cd79dc78050)
--------------------------------------------------------------------------

We replaced the values that had smaller representation and combined them into "other" categories.
We leveraged pd.get_dummies to convert all of the categorical data so that it can be consumed by the models.

### Test Train Split and the Model

![image](https://github.com/SwagmasterMatt/deep-learning-challenge/assets/133460903/ecbd2aaa-4a4f-4880-acc8-7ce185af19e5)
We used a standard scaler to normalize the data so that the model did not become bias.

3 layers - 100 epochs
1. 80 - relu
2. 30 - relu
3. 1 - signmoid

#### The result: 
--------------------------------------------------------------------------
![image](https://github.com/SwagmasterMatt/deep-learning-challenge/assets/133460903/2dac415c-a4b7-46fd-b054-a2e79f9eb3c3)
--------------------------------------------------------------------------




## Alphabet Soup Charity Optimization

Purpose - In this run we are trying to optimize greater than our first attempt which stalemated at 73% accuracy.

Our target variable is still the IS_SUCCESSFUL binary 

Previously our features excluded the NAME column. This time we are including it for a little more data and noise in the model to tackle the overfitting that was occuring in the first model.

Our total features now are: 
- NAME—Identification
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special considerations for application
- ASK_AMT—Funding amount requested

Exclusions:
-----------------------
- EIN
We're excluding EIN because it is too unique and does not share the same control in noise as the name column. This makes EIN neither a target or feature.

### Additional Pre Processing
To enhance the noise of the model and gain a better understanding of the distribution of numbers we conducted value counts of ASK_AMT and NAME.

![image](https://github.com/SwagmasterMatt/deep-learning-challenge/assets/133460903/18224520-e513-47ee-b7e3-d1bc333848e5)
----------------------------------------------------------------------------

We apply a logarithmic manipulation to the ASK_AMT due to the majority of values residing with the amount of 5000.

We applied similar binning methdos to the Name value counts. We decided to go with a 1/2 Percent of the largest amount so that the data set remained a minority of the count. This is what we use to control the level of our noise.

### Keras Tuner

We used a Keras Tuner to optimize the model further. We put in several optimization choices for the model:

1. Dropout - to prevent overfitting
2. Kernel_regularizer - to prevent overfitting and fight higher weights
3. Choice of Activation Methods - Relu and Tanh were the two activation methods that best fit the data
----------------------------------------------------------------------------
![image](https://github.com/SwagmasterMatt/deep-learning-challenge/assets/133460903/b6cbcd60-8629-4760-b5c5-6b643bbf23f9)
----------------------------------------------------------------------------
4. Early Stopping - This prevented the model from being overtrained on the test data once performance in the validation accuracy started to decline.
5. Floor Callback - performance_threshold allowed us to not waste resources if the accuracy started well below our original model. We chose 72%.
6. Accuracy Threshold - Initially we had a threshold of 76% to quickly run through iterations of testing. After successful methods were established we raised the threshold to 85% to act as a ceiling.

### Tuner Comparison 
----------------------------------------------------------------------------
![image](https://github.com/SwagmasterMatt/deep-learning-challenge/assets/133460903/53324804-0049-49f3-aebb-3ea265a48a6b)
----------------------------------------------------------------------------
We attempted 2 different optimization methods
- Hyperband
- Bayesian Optimization

Initially, with less than 25 epochs, the hyperband out performed the bayesian. 
----------------------------------------------------------------------------
![image](https://github.com/SwagmasterMatt/deep-learning-challenge/assets/133460903/e02bdae7-53e5-400e-a83a-dbeb17925d7f)
----------------------------------------------------------------------------
#### HyperBand Accuracy over epochs
![image](https://github.com/SwagmasterMatt/deep-learning-challenge/assets/133460903/12e0d9be-d39b-4db3-841a-d4deb4cafeaa)

Early Stopping was implemented

![image](https://github.com/SwagmasterMatt/deep-learning-challenge/assets/133460903/6a158189-9964-4f00-b4fa-c597fc474ed8)

Accuracy

![image](https://github.com/SwagmasterMatt/deep-learning-challenge/assets/133460903/c0ccc167-c4af-4f11-87b6-79bdd7ed4ac1)


#### Bayesian Optimization over epochs
![image](https://github.com/SwagmasterMatt/deep-learning-challenge/assets/133460903/0283640d-4687-4bbe-9cab-bad4bac5b370)

Early Stopping was implemented 

![image](https://github.com/SwagmasterMatt/deep-learning-challenge/assets/133460903/11def60f-28cc-4367-80a7-a16f9fa6acaa)

Accuracy

![image](https://github.com/SwagmasterMatt/deep-learning-challenge/assets/133460903/3c117990-e602-4487-93d1-ec1d0020f475)


#### Attempt Ensemble Optimization

Leveraging the best of both models, we could resonably attempt to include the average of the results of the top 3 from each tuner attempt. Ensemble accuracy: 0.7836734693877551

![image](https://github.com/SwagmasterMatt/deep-learning-challenge/assets/133460903/c6130463-b255-4ddc-aa6b-f630a99bb091)

Several warnings came with the Ensemble methods mostly due to less compatibility between scikit learn and tensorflow for deep learning neural networks.

Ultimately we chose to save the Bayesian Model with a Loss: 0.48945945501327515 & Accuracy: 0.7810495495796204 once it was given additional epochs and an additional accuracy test. Either result would have been acceptable.

## Alternative Methods & Summary

Either model tuner method could have been chosen to complete the model. RandomSearch is also avaialble as a tuner. Additonally, randomtrees could be used to do feature selection and further enhance the pre processing as well as keras_tuner preprocessing. We were able to meet the criteria of getting a model above 75%. The main problem we had to tackle during optimization was overfitting and chose to increase noise in the data set by normalizng the name field. Tensorflow also has methods for generating noise in a data set. Advancement of these models would be seeking to make them more resource efficient by not choosing multiple keras tuners and balancing load and learning speed as part of optimization to determine the fastest way to train the models. 

The Keras Tuners produced our best parameters for neurons, layers, and activation functions. The initial and acceptable range was seleced due to trial and error performance after over 100 runs of different tuner variants. 

### Best Hyperband parameters

Hyperband values: {'units': 180, 'dropout': 0.35000000000000003, 'num_layers': 2, 'units_0': 36, 'activation_0': 'relu', 'units_1': 6, 'activation_1': 'relu', 'units_2': 41, 'activation_2': 'tanh', 'units_3': 31, 'activation_3': 'tanh', 'units_4': 11, 'activation_4': 'relu', 'tuner/epochs': 25, 'tuner/initial_epoch': 9, 'tuner/bracket': 2, 'tuner/round': 2, 'tuner/trial_id': '0013'}

### Best Bayesian parameters

Bayesian values: {'units': 60, 'dropout': 0.05, 'num_layers': 5, 'units_0': 21, 'activation_0': 'relu', 'units_1': 76, 'activation_1': 'relu', 'units_2': 56, 'activation_2': 'tanh', 'units_3': 6, 'activation_3': 'tanh', 'units_4': 86, 'activation_4': 'relu'}
