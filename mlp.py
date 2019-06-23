import keras
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Activation, Dense
from keras.models import Sequential
from keras import optimizers
from keras.callbacks import Callback
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import argparse
import locale
import os	


df = pd.read_csv("https://aisgaiap.blob.core.windows.net/aiap4-assessment/real_estate.csv")
df = pd.DataFrame(df)
print(df.head())
prices = df['Y house price of unit area']
features = df.drop('Y house price of unit area', axis = 1)
    

print("Taipei housing dataset has {} data points with {} variables each.".format(*df.shape))

# Minimum price of the data
minimum_price = np.amin(prices)

# Maximum price of the data
maximum_price = np.amax(prices)

# Mean price of the data
mean_price = np.mean(prices)

# Median price of the data
median_price = np.median(prices)

# Standard deviation of prices of the data
std_price = np.std(prices)

# Show the calculated statistics
print("Statistics for Taipei housing dataset:\n")
print("Minimum price: ${}".format(minimum_price)) 
print("Maximum price: ${}".format(maximum_price))
print("Mean price: ${}".format(mean_price))
print("Median price ${}".format(median_price))
print("Standard deviation of prices: ${}".format(std_price))

#sns.pairplot(df, height=2.5)
#plt.tight_layout()

dataset = df.values
#print(dataset)

X = dataset[:,2:6]
Y = dataset[:,7]

min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
Y_scale = Y/maximum_price


X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y_scale, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

#print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)

model = Sequential([
    Dense(32, activation='relu', input_shape=(4,)),
    Dense(16, activation='relu'),
    Dense(1, activation='linear'),
])

opt = keras.optimizers.Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(optimizer=opt,
              loss='mse')
hist = model.fit(X_train, Y_train,
          batch_size=8, epochs=100,
          validation_data=(X_val, Y_val))
model.evaluate(X_test, Y_test)



# make predictions on the testing data
print("[INFO] predicting house prices...")
preds = model.predict(X_test)
 
# compute the difference between the *predicted* house prices and the
# *actual* house prices, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - Y_test
percentDiff = (diff / Y_test) * 100
absPercentDiff = np.abs(percentDiff)
 
# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)
 
# finally, show some statistics on our model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. house price: {}, std house price: {}".format(
	locale.currency(df["Y house price of unit area"].mean(), grouping=True),
	locale.currency(df["Y house price of unit area"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))
