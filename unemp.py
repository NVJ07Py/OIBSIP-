import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

# Load the data
df = pd.read_csv("D:/hko/Unempl.csv")

# Display the first few rows of the DataFrame
print(df.head())
print(df.describe())


# Check for missing values
missing_values = df.isnull().sum()
print("Missing values:\n", missing_values)

# Check value counts for the 'Region' column
print(df['Region'].value_counts())

sns.pairplot(df)
plt.show()
df['Region'].value_counts().plot.pie()
plt.show()
df['Area'].value_counts().plot.bar()
plt.show()


# Assuming df contains the relevant columns
df.columns = ["Region", "Date", "Frequency", "Unemployment_Rate", "Estimated_Employed", "Labour_Participation_Rate", "Area"]

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="Estimated_Employed", hue="Area", kde=True, palette="Set2")
plt.title("Histogram of` Estimated Employment Rate by Area")
plt.xlabel("Estimated_Employed")
plt.ylabel("Count")
plt.show()

