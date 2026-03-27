import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.title("Using machine learning to predict heart disease")
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


train.info()
st.dataframe(train.head(5))


train['Heart Disease'] = train['Heart Disease'].map({'Presence': 0, 'Absence': 1})


matrix = train.corr()
st.dataframe(matrix)


train = train.drop('id', axis='columns')
train.hist(bins=50, figsize=(20,15))
st.pyplot(plt)

variables = train.columns.tolist()
st.sidebar.title("Variables")
selected_variable = st.sidebar.selectbox("Select a Variable:", variables)
st.subheader(f"Distribution of {selected_variable}")
fig, ax = plt.subplots()
ax.hist(train[selected_variable], bins=30, edgecolor='black')
ax.set_xlabel(selected_variable)
ax.set_ylabel("Count")
st.pyplot(fig)


matrix["Age"].sort_values(ascending=False)



train['exercise_angina_by_age'] = train['Exercise angina'] / train['Sex']



train['cholesterol_blood_pressure'] = train['Cholesterol'] / train['BP']


matrix = train.corr()
matrix['Age'].sort_values(ascending=False)


train.info()


train_numerical = train.drop(['ST depression', 'exercise_angina_by_age', 'cholesterol_blood_pressure'], axis=1)


from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler(feature_range=(-1,1))
train_numerical_min_max_scaled = min_max_scaler.fit_transform(train_numerical)



from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
train_numerical_std_scaled = std_scaler.fit_transform(train_numerical)


from sklearn.metrics.pairwise import rbf_kernel

age_simil_35 = rbf_kernel(train[['Age']], [[35]], gamma=0.1)
print(age_simil_35)


from sklearn.linear_model import LinearRegression

target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(train_numerical)

model = LinearRegression()
model.fit(train_numerical, scaled_labels)
scaled_predictions = model.predict(train_numerical)
predictions = target_scaler.inverse_transform(scaled_predictions)

print(predictions)


from sklearn.compose import TransformedTargetRegressor

model = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())

model.fit(train_numerical, train['Heart Disease'])
predictions = model.predict(train_numerical)

print(predictions)


from sklearn.preprocessing import FunctionTransformer
log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_cholesterol = log_transformer.transform(train[['Cholesterol']])
print(log_cholesterol)


from sklearn.model_selection import train_test_split
X = train.drop(['Heart Disease', 'exercise_angina_by_age'],  axis=1)
y = train['Heart Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


X.info()


X['cholesterol_blood_pressure'] = X['cholesterol_blood_pressure'].astype('int64')


X['ST depression'] = X['ST depression'].astype('int64')


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

log_reg = make_pipeline(StandardScaler(), LogisticRegression())
log_reg.fit(X_train, y_train)
heart_disease_predictions = log_reg.predict(X_train)
print(heart_disease_predictions)


from sklearn.metrics import mean_squared_error
log_rmse = mean_squared_error(y_train, heart_disease_predictions)
log_rmse

st.metric(label="RMSE for Logistic Regression Model", value=f'{log_rmse:.4f}', delta=None)

from sklearn.tree import DecisionTreeClassifier

tree_classifier = make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=42))
tree_classifier.fit(X_train, y_train)
heart_disease_pred_tree = tree_classifier.predict(X_train)
print(heart_disease_pred_tree)


tree_rmse = mean_squared_error(y_train, heart_disease_pred_tree)
print(tree_rmse)


from sklearn.model_selection import cross_val_score
tree_rmses = -cross_val_score(tree_classifier, X_train, y_train, scoring='neg_root_mean_squared_error', cv=10)
pd.Series(tree_rmses).describe()


from sklearn.ensemble import RandomForestClassifier

forest_classifier = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))
forest_rmses = -cross_val_score(forest_classifier, X_train, y_train, scoring='neg_root_mean_squared_error', cv=10)


pd.Series(forest_rmses).describe()





