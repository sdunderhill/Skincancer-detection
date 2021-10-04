#!/usr/bin/env python
# coding: utf-8

# # Preprocessing the Data for a Neural Network

# Import our dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import pandas as pd
import pickle
# import tensorflow as tf

#  Import and read the HAM10000_metadata.csv.
import pandas as pd 
metadata_df = pd.read_csv("resources/HAM10000_metadata.csv")
metadata_df.head()


# Drop the non-beneficial ID columns
metadata_df = metadata_df.drop(['dx_type'],axis=1)
metadata_df.head()


# Determine the number of unique values in each column.
metadata_df.nunique()


# Look at dx counts for binning
dx_counts = metadata_df['dx'].value_counts()
dx_counts


# Combine mel and bcc categories, rename as 'cancer'
metadata_df['dx'] = metadata_df['dx'].replace({'mel': 'Cancer', 'bcc': 'Cancer', 'nv': 'Mole'})
metadata_df

# Look at dx counts for binning
dx_counts = metadata_df['dx'].value_counts()
dx_counts

# Determine which values to replace if counts are less than ...?
replace_dx = list(dx_counts[dx_counts < 1500].index)

# Replace in dataframe
for dx in replace_dx:
    metadata_df.dx = metadata_df.dx.replace(dx,"Other")
    
# Check to make sure binning was successful
metadata_df.dx.value_counts()

# Visualize the value counts of APPLICATION_TYPE
dx_counts.plot.density()

# Look at localization counts for binning
localization_counts = metadata_df['localization'].value_counts()
localization_counts

# Determine which values to replace if counts are less than ...?
replace_localization = list(localization_counts[localization_counts < 700].index)

# Replace in dataframe
for localization in replace_localization:
    metadata_df.localization = metadata_df.localization.replace(localization,"Other")
    
# Check to make sure binning was successful
metadata_df.localization.value_counts()

# Visualize the value counts of APPLICATION_TYPE
localization_counts.plot.density()

# Export cleaned csv
metadata_df.to_csv('metadata_filtered.csv')

# Generate our categorical variable lists
metadata_cat = metadata_df.dtypes[metadata_df.dtypes == "object"].index.tolist()
metadata_cat


# Create a OneHotEncoder instance
enc = OneHotEncoder(sparse=False)

# Fit and transform the OneHotEncoder using the categorical variable list
encode_df = pd.DataFrame(enc.fit_transform(metadata_df[metadata_cat]))

# Add the encoded variable names to the dataframe
encode_df.columns = enc.get_feature_names(metadata_cat)
encode_df.head()

# Merge one-hot encoded features and drop the originals
metadata_df = metadata_df.merge(encode_df,left_index=True, right_index=True)
metadata_df = metadata_df.drop(metadata_cat,axis=1)
metadata_df.head()


# Split our preprocessed data into our features and target arrays
y = metadata_df['dx_Cancer'].values
X = metadata_df.drop(['dx_Cancer'],1).values

# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78,stratify=y)


# Create a StandardScaler instances
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# # Compile, Train, and Evaluate the Model

import tensorflow as tf

# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.

nn = tf.keras.models.Sequential()
number_input_features = len(X_train[0])
hidden_nodes_layer1 =  9
hidden_nodes_layer2 = 3

# First hidden layer
nn.add(
    tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu")
)

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="relu"))

# Check the structure of the model
nn.summary()

# Compile the model
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
fit_model = nn.fit(X_train_scaled, y_train, epochs=100)

# Evaluate the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

# Export our model to HDF5 file
nn.save("h5/CancerIdentification.h5")

nn_imported = tf.keras.models.load_model('h5/CancerIdentification.h5')

# Evaluate the completed model using the test data
model_loss, model_accuracy = nn_imported.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

# Saving model to disk
pickle.dump(fit_model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))