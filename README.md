# HPthingy

## File name is mlp.py

In this code, I am using standard linear regression (Multi-layer perceptron model) from the keras library to run the machine learning algorithm

The first 47 lines of code are to import the relevant packages and display the information using pandas. So I am able to get the House Price of Unit Area, and the statistics of the Prices, mean, median, max, min, etc...

### Lines 53-54
I am taking 4 inputs: Distance to MRT, Number of convenience stores, Long, Lat

###Lines 56-58
Using the Scikit learn library, I am now constraining the input features, so that they lie within 0 and 1
Same goes for the Y output, just divide by the max price

### Lines 61-62
Splitting the datasets into train sets, test and validation sets
Due to the constraints of the scikitlearn library, it has to be done twice.
Basically
One Dataset -> train + test/val
test/val -> test + val

### Lines 66 - 70
This is a MLP model using the keras library
4 inputs -> 32 neurons -> 16 neurons -> 1 output

### Lines 72 - 74
Optimizer - ADAM
Loss-Mean Squared Error

### Lines 75 - 78
Training and Evaluation

### Lines 82-103
Found this part online to do comparision for the results
To find the percentage off in the housing price predictions with a standard deviation of a certain percentage also
