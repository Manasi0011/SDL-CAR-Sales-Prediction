import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

car = pd.read_csv("car data (1).csv")

car['car_age'] = 2020 - car['Year']
car.drop('Year', axis=1, inplace=True)

fuel = pd.get_dummies(car['Fuel_Type'])
transmission = pd.get_dummies(car['Transmission'], drop_first=True)
seller = pd.get_dummies(car['Seller_Type'], drop_first=True)

fuel.drop('CNG', axis=1, inplace=True)

car = pd.concat([car, fuel, transmission, seller], axis=1)

car.drop(['Fuel_Type', 'Seller_Type', 'Transmission'], axis=1, inplace=True)

car = car.drop('Car_Name', axis=1)

new_car = car.drop(['company', 'Petrol', 'Individual', 'Kms_Driven'], axis='columns')

x_data = new_car.drop(['Selling_Price'], axis='columns')
y_data = new_car['Selling_Price']

linear_reg = LinearRegression()

linear_reg.fit(x_data, y_data)

linear_reg.predict([[6.9, 1, 1, 0, 1]])

linear_reg.score(x_data, y_data)

pickle.dump(linear_reg, open("model.pkl", "wb"))
model = pickle.load(open("model.pkl", "rb"))
