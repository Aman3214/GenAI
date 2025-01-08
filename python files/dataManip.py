import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
file_path = 'data/data.csv'  
data = pd.read_csv(file_path)

# converting yr_built and yr_renovated to how old the property is 
coulmns_to_simplify = ['yr_built', 'yr_renovated']
for column in coulmns_to_simplify:
    max_date = data[column].max()
    data[column] = (max_date - data[column])/max_date * 10

# normalizing the prices,sqft_living,sqft_lot,sqft_above,sqft_basement,bedrooms,bathrooms
scaler = MinMaxScaler()
columns_to_normalize = ['price', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'bedrooms', 'bathrooms']
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

#encoding city and statezipcode
encoder = LabelEncoder()
data['city'] = encoder.fit_transform(data['city'])
data['statezip'] = encoder.fit_transform(data['statezip'])

#removing unecessary data
data.drop(columns=['street','country','date'], inplace=True)

#rounding and updating the data
data = data.round(3)
data.to_csv('data/updated_data.csv', index=False)

# splitting the data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
train_data.to_csv('data/train.csv', index=False)
test_data.to_csv('data/test.csv', index=False)
