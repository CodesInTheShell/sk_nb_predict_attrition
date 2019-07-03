import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
import pickle

print('---------------- DATA PREPARATION --------------')
df = pd.read_csv('attrition_2_cols.csv')
# print(df.head())

inputs = df.drop('Attrition', axis='columns')
target = df['Attrition']
# print(inputs.head())
# print(target.head())

le_target = LabelEncoder()
le_gender = LabelEncoder()
le_marital = LabelEncoder()

n_target = le_target.fit_transform(target)
le_target_mapping = dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))
print('Saving target label encoder as le_target.pkl')
with open('le_target.pkl', 'wb') as file:  
	pickle.dump(le_target, file)

inputs['n_Gender'] = le_gender.fit_transform(inputs['Gender'])
le_gender_mapping = dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))
print('Saving gender label encoder as le_gender.pkl')
with open('le_gender.pkl', 'wb') as file:  
	pickle.dump(le_gender, file)

inputs['n_MaritalStatus'] = le_marital.fit_transform(inputs['MaritalStatus'])
le_marital_mapping = dict(zip(le_marital.classes_, le_marital.transform(le_marital.classes_)))
print('Saving marital label encoder as le_marital.pkl')
with open('le_marital.pkl', 'wb') as file:  
	pickle.dump(le_marital, file)

n_inputs = inputs.drop(['Gender', 'MaritalStatus'], axis='columns')
# print(n_inputs.head())

print('---------------------- LABEL MAPPING -------------------------')
print('Label map of target', le_target_mapping)
print('Label map of gender', le_gender_mapping)
print('Label map of marital', le_marital_mapping)

# --- TRAINING THE MODEL ---
print('-------------------- TRAINING THE MODEL ------------------------')
model = GaussianNB() #MultinomialNB() #BernoulliNB() #GaussianNB() 
model.fit(n_inputs,n_target)

with open('model.pkl', 'wb') as file:  
	pickle.dump(model, file)
print('Model saved as model.pkl that can be use for future predictions without the need of retraining the model.')

# ------ MODEL SCORE --------------
model.score(n_inputs, n_target)
print("Score: ", model.score(n_inputs, n_target))

# ------ PREDICTING NEW DATA --------------
print('------------------- PREDICTING NEW DATA ------------------------')
# Load the model
model_pkl = open('model.pkl', 'rb')
model = pickle.load(model_pkl)
model_pkl.close()

# Load the encoders
le_target_pkl = open('le_target.pkl', 'rb')
le_target = pickle.load(le_target_pkl)
le_target_pkl.close()

le_gende_pkl = open('le_gender.pkl', 'rb')
le_gende = pickle.load(le_gende_pkl)
le_gende_pkl.close()

le_marital_pkl = open('le_marital.pkl', 'rb')
le_marital = pickle.load(le_marital_pkl)
le_marital_pkl.close()

# Predict new data
new_data = [['Female'], ['Single']]
sample_data = [[le_gender.transform(['Female'])[0], le_marital.transform(['Single'])[0]]]

print('Class predicted: ', list(le_target.inverse_transform(model.predict(sample_data))))
# print(list(le_target.inverse_transform(model.predict(sample_data))))
print('\n')
print('Probabilities on both classes: ')
print('\tClass and Label: ', le_target_mapping)
print('\tProbabilities: ', model.predict_proba(sample_data))

print('------------------- DONE -----------------------------')