# %%
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import plotly.express as px
import pickle

# %%
# split input in dialog act and sentence
split_lines = []
with open('dialog_acts.dat', 'r') as f:  
    for line in f:
        split_lines.append(line.rstrip().split(' ', 1))
    
data = pd.DataFrame(split_lines, columns=['dialog_act', 'sentence'])

data.head()

# %%
# Split input into training and test data

x_train, x_test, y_train, y_test = train_test_split( data["sentence"], data["dialog_act"] , test_size=0.15, random_state=42)

# %%
# A simple baseline model that always predicts "inform"

def base_line1(data):
    return "inform"

# %%
#print the most comman sentences in the training data
#print(data.value_counts().to_string())

# %%
# A second baseline model that uses a set of rules to predict the dialog act
# The rules are based on the words in the sentence
# TODO: Improve the rules to increase the accuracy of the model

rules = {
    'bye': ['bye', 'goodbye', 'see you', ],
    'affirm': ['yes', "yep",'right', 'do that', 'agreed', 'yeah', 'yea', 'correct'],
    'repeat': ['repeat'],
    'reqalts': ['else', 'anything', 'about', 'another', 'next', 'other'],
    'reqmore': ['more'],
    'request': ['address', 'post', 'what', 'postal', 'phone', 'number', 'whats', 'where', 'venue', 'give', 'located',  'telephone'],
    'ack': ['uhm', 'fine', 'sure'],
    'confirm': ['is it', 'does it'],
    'deny': ['not', 'wrong'],
    'hello': ['hello'],
    'negate': ["nope" , "not that" ],
    'null': ['cough', 'noise', 'sil', 'unintelligible', 'breathing', 'inaudible'],
    'restart': [ 'start', 'restart', 'again'],
    'thankyou': ['thank', 'thanks', 'goodbye'],
    'inform': [
        'looking', 'any', 'matter', 'care', 'north', 'west', 'east', 'south', 'european', 'italian', 'korean',
        'food', 'moderate', 'cheap', 'expensive', 'restaurant', 'town', 'part',  'find', 'center',
    ]
}






def base_line2(x):
    for key, value in rules.items():
        for v in value:
            if v in x:
                return key
    return "inform"

# %%
# Evaluate the baseline models
y_baseline1 = x_test.apply(base_line1)
y_baseline2 = x_test.apply(base_line2)

print("Baseline 1:")
print(classification_report(y_test, y_baseline1))

print("Baseline 2:")
print(classification_report(y_test, y_baseline2))

# %%
# Dedupe the data for the machine learning models
deduped_data = data.drop_duplicates(subset='sentence')

# Split the deduped data into training and test data

x_train_deduped, x_test_deduped, y_train_deduped, y_test_deduped = train_test_split( deduped_data["sentence"], deduped_data["dialog_act"] , test_size=0.15, random_state=42)
x_train_dupe, x_test_dupe, y_train_dupe, y_test_dupe = train_test_split( data["sentence"], data["dialog_act"] , test_size=0.15, random_state=42)

# %%
# Use a simple bag of words model to vectorize the input data
count_vectorizer = CountVectorizer()
x_train_count = count_vectorizer.fit_transform(x_train_deduped)

# Save 
pickle.dump(count_vectorizer, open("classifiers/count_vectorizer.pkl", "wb"))

# Create a visual representation of the data using PCA
# this can be used to get a feeling for the data and to see if the data is separable

pca = PCA(n_components=3)
x_train_pca = pca.fit_transform(x_train_count.toarray())

fig = px.scatter_3d(x=x_train_pca[:,0], y=x_train_pca[:,1], z=x_train_pca[:,2], color=y_train_deduped)

fig.show()

# %%
# do the pca with the duped data
count_vectorizer_dupe = CountVectorizer()
x_train_count_dupe = count_vectorizer_dupe.fit_transform(x_train_dupe)

pickle.dump(count_vectorizer_dupe, open("classifiers/count_vectorizer_duped.pkl", "wb"))


pca = PCA(n_components=3)

x_train_pca_dupe = pca.fit_transform(x_train_count_dupe.toarray())

fig = px.scatter_3d(x=x_train_pca_dupe[:,0], y=x_train_pca_dupe[:,1], z=x_train_pca_dupe[:,2], color=y_train_dupe)

fig.show()

# %%
#compare the len on non deduped and deduped data
print("non deduped len:",len(data))
print("deduped len:",len(deduped_data))

# %%
# Train a logistic regression model on the data

logistic_regression_model = LogisticRegression()

logistic_regression_model.fit(x_train_count, y_train_deduped)

x_test_count = count_vectorizer.transform(x_test_deduped)

y_pred = logistic_regression_model.predict(x_test_count)

print("Logistic Regression, non deduped  data   " )
print(classification_report(y_test_deduped, y_pred))
pickle.dump(logistic_regression_model, open("classifiers/logistic_regression_deduped.pkl", "wb"))

# %%
# Train a random forest model on the data

random_forest_model = RandomForestClassifier()

random_forest_model.fit(x_train_count, y_train_deduped)

x_test_count = count_vectorizer.transform(x_test_deduped)

y_pred = random_forest_model.predict(x_test_count)

print("Random Forest, non deduped data   ")
print(classification_report(y_test_deduped, y_pred))
pickle.dump(random_forest_model, open("classifiers/random_forest_deduped.pkl", "wb"))

# %%
# train a knn model on the data 

knn_model = KNeighborsClassifier(n_neighbors=5, metric="manhattan"  , algorithm="kd_tree"  , n_jobs=None)
 
knn_model.fit(x_train_count, y_train_deduped)

x_test_count = count_vectorizer.transform(x_test_deduped)

y_pred = knn_model.predict(x_test_count)

print("KNN, non deduped data")
print(classification_report(y_test_deduped, y_pred))
pickle.dump(knn_model, open("classifiers/knn_deduped.pkl", "wb"))


# %%
# Train a logistic regression model on the duped data

logistic_regression_model = LogisticRegression()

logistic_regression_model.fit(x_train_count_dupe, y_train_dupe)

x_test_count = count_vectorizer_dupe.transform(x_test_dupe)

y_pred = logistic_regression_model.predict(x_test_count)

print("Logistic Regression, deduped data")
print(classification_report(y_test_dupe, y_pred))
pickle.dump(logistic_regression_model, open("classifiers/logistic_regression_duped.pkl", "wb"))


# %%
# Train a random forest model on the duped data

random_forest_model = RandomForestClassifier()

random_forest_model.fit(x_train_count_dupe, y_train_dupe)

x_test_count = count_vectorizer_dupe.transform(x_test_dupe)

y_pred = random_forest_model.predict(x_test_count)

print("Random Forest, deduped data")
print(classification_report(y_test_dupe, y_pred))
pickle.dump(random_forest_model, open("classifiers/random_forest_duped.pkl", "wb"))


# %%
# train a knn model on teh duped data  

knn_model = KNeighborsClassifier(n_neighbors=3, metric="manhattan"  , algorithm="kd_tree"  , n_jobs=None)

knn_model.fit(x_train_count_dupe, y_train_dupe)

x_test_count = count_vectorizer_dupe.transform(x_test_dupe)

y_pred = knn_model.predict(x_test_count)


print(classification_report(y_test_dupe, y_pred))
pickle.dump(knn_model, open("classifiers/knn_duped.pkl", "wb"))


# %%







# %%

# %% Use a simple bag of words model to vectorize the input data
count_vectorizer = CountVectorizer()

# Fit the vectorizer on the training data
x_train_count = count_vectorizer.fit_transform(x_train_deduped)

# Save the fitted vectorizer for reuse
pickle.dump(count_vectorizer, open("classifiers/count_vectorizer.pkl", "wb"))

# Transform the test data using the fitted vectorizer
x_test_count_deduped = count_vectorizer.transform(x_test_deduped)

# Create a visual representation of the training data using PCA
pca = PCA(n_components=3)
x_train_pca = pca.fit_transform(x_train_count.toarray())

fig = px.scatter_3d(x=x_train_pca[:, 0], y=x_train_pca[:, 1], z=x_train_pca[:, 2], color=y_train_deduped)
fig.show()

# %% Train a logistic regression model on the data
logistic_regression_model = LogisticRegression()

# Fit the logistic regression model
logistic_regression_model.fit(x_train_count, y_train_deduped)

# Make predictions on the test set
y_pred_deduped = logistic_regression_model.predict(x_test_count_deduped)

# Print classification report
print("Logistic Regression, deduped data")
print(classification_report(y_test_deduped, y_pred_deduped))

# Save the model
pickle.dump(logistic_regression_model, open("classifiers/logistic_regression_deduped.pkl", "wb"))

# %% Identify and print the misclassified instances for the deduped Logistic Regression model
# Find the indexes of the misclassified instances
misclassified_indices = [i for i in range(len(y_test_deduped)) if y_test_deduped.iloc[i] != y_pred_deduped[i]]

# Create a DataFrame containing the misclassified instances
misclassified_cases = pd.DataFrame({
    'sentence': x_test_deduped.iloc[misclassified_indices].values,
    'true_label': y_test_deduped.iloc[misclassified_indices].values,
    'predicted_label': y_pred_deduped[misclassified_indices]
})

# Print the misclassified cases to the console
print("Misclassified Cases by Logistic Regression (deduped data):")
print(misclassified_cases.to_string(index=False))


