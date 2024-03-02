import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv("D:/hko/Advertising.csv")

# Display the first few rows of the DataFrame
print(df.head())
print(df.describe())

# Create a boxplot to show the sales 5 number series 
print("box plot")
plt.boxplot(df["Sales"])
plt.show()

df.isnull().sum()
df[df.duplicated()]

sns.pairplot(df)
plt.show()

sns.pairplot(data = df , hue = 'Sales')
plt.show()


# Split the dataset into training and testing sets
from sklearn.ensemble import RandomForestRegressor

X = df.drop(['Sales'], axis=1)
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

forest = RandomForestRegressor(n_estimators=10)
forest.fit(X_train, y_train)

predictions = forest.predict(X_test)

print("Mean Squared Error: ", np.mean((predictions - y_test) ** 2))

# Visualizing the importance of features in the Random Forest Model
importances = forest.feature_importances_
plt.bar(X.columns, importances)
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.show()

