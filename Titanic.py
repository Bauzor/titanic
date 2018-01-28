#   Hi, This is my DeltaHacks Project, I am a one man team just looking to spend
# 24 hours watching a crap ton of data visualization and data science tools and
# tricks that I'll be able to get better at and learn for the future to
# eventually be able to do predictive modelling. I do not have enough knowledge
# to make a legitimate application but this is my effort to work towards that
# goal as this is the first hackathon I have attended in which I can somewhat
# say that I can "code" Enjoy!
#
#                                                   ~ Bryan Au
#===============================================================================
#
# imports for data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
#
# imports for visualization
import seaborn as sns
import matplotlib.pyplot as plt
#
# imports for modelling and machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
#
#===============================================================================
#
#Function definitions that will be used for Data Visualization
#
#Plots a continuous graph of a data set vs a target
def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()
#
#Compares all fields against eachother to look for correlation
def plot_correlation_map( df ):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr,
        cmap = cmap,
        square=True,
        cbar_kws={ 'shrink' : .9 },
        ax=ax,
        annot = True,
        annot_kws = { 'fontsize' : 12 }
    )
#
#===============================================================================
#
#File paths for CSV
test_file_path = 'C:/Users/Bryan/Documents/Comp Sci/Python Hype Train/Data_Science/test.csv'
train_file_path = 'C:/Users/Bryan/Documents/Comp Sci/Python Hype Train/Data_Science/train.csv'
#
#Reading CSV files
test_data = pd.read_csv(test_file_path)
df = pd.read_csv(train_file_path)
#
# --------- Sorting training data into respective columns --------- #
#Mainly used to remember the columns for reference
#PassengerID(index)
df_PassengerId = df.PassengerId
#Target Variable
df_Survived = df.Survived
#1 == first class, 2 == second class, 3 == third class
df_Pclass = df.Pclass
#Name of all the passengers, pretty useless
df_Name = df.Name
# Sex is important (Woman board before men)
df_Sex = df.Sex
# Would probably leave the old to die
df_Age = df.Age
#number of siblings are spouses abord ship
df_SibSp = df.SibSp
#number of parents or children aboard ship
df_Parch = df.Parch
#ticket_number
df_Ticket = df.Ticket
#How much a person paid for their ticket
df_Fare = df.Fare
#Location of their cabin
df_Cabin = df.Cabin
#C = Cherbourg, Q = Queenstown, S = Southapmton
df_Embarked = df.Embarked
#variables that may or may not effect whether or not a person survived the wreck
could_effect_Survived = ["Pclass","Sex","Age","SibSp","Parch","Cabin","Embarked"]
#Training data that I think is the best from the above
data_predictors = df[could_effect_Survived]
#
#--------------------- Single Variable Analysis -------------------------------#
#Used to check if there are missing fields in the data set
print(df.count())
#Notice we have 177 missing Ages
# 2 Missing embarked locations
# and we're missing 687 Cabin rooms
# too little cabin data so it might be best if we omit it from the set but
# researhcing is a better option
# the missing ages can be computed using either the average age or if i was good
# enough, a machine learning algorithm
#Oldest Youngest
print("Gathering some basic info from the original failure:\n")
print("The youngest person aboard the ship was %f and the oldest was %f\n\n"%
(df_Age.min(),df_Age.max()))
print(df.Age.mean())

# Number of Survived
print("The number of people who survived and die in the shipwreck\n")
print(df["Survived"].value_counts())
print("\n\n")
# Percent of people that survived
print("Percentage of victims, and survivors")
print(df_Survived.value_counts() * 100 / len(df))
#Check if there's a large discrepancy between max and min of Parent childs
print("The max amount of children and or parents was %d"%(df_Parch.min()))
print("The min amount of childre and or parents was %d"%(df_Parch.max()))
#Check the amount of males and females that were aboard the titanic
print("The amount of males and females were:")
print(df_Sex.value_counts())
#Check the amount of people that got on from each Location
print("Count of where most people Embarked from")
print(df_Embarked.value_counts().sort_index())
#Num of Each class
print("Count of each class")
print(df_Pclass.value_counts().sort_index())


#Showing off some of the data
print("This is the test data (first few rows): ")
print(test_data.head())
print("\n")
print("This is the training data summary: ")
print(df.head())
print("\n")



#Showing off some of the data summary
print("This is the test data summary: ")
print(test_data.describe())
print("\n")
print("This is the training data summary: ")
print(df.describe())
print("\n")

print("\n")
print("These are the column headers of the training data: ")
print(df.columns)
print("These are the column headers of the test data: ")
print(test_data.columns)
print("\n")
print("Data that might've affected survival rate summary: ")
print(data_predictors.describe())



print("\n")
print("These are the column headers: ")
print(df.columns)
print("\n")
print("Data that might've affected survival rate (first few rows): ")
print(data_predictors.head())

#Choosing Prediction Target
print(df.sample(5))

plot_correlation_map(df)
plt.show()

bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]

df["Agebins"] = pd.cut(df["Age"], bins)

df[df["Survived"] == 1]["Agebins"].value_counts().sort_index().plot(kind="bar")
plt.show()

df[df["Survived"] == 0]["Agebins"].value_counts().sort_index().plot(kind="bar")
plt.show()

df["Agebins"].value_counts().sort_index().plot(kind="bar")
plt.show()

sns.barplot(x="Pclass", y="Survived", hue="Sex", data=df);
plt.show()

plot_distribution(df[df["Survived"] == 0], var = 'Age' , target = 'Survived' , row = 'Sex' )
plt.show()

plot_distribution(df[df["Survived"] == 1], var = 'Age' , target = 'Survived' , row = 'Sex' )
plt.show()

plot_distribution(df[(df.Survived) == 1 & (df.Sex == "female")], var = 'Age' , target = 'Survived' , row = 'Sex' )
plt.show()

plot_distribution(df[(df.Survived) == 0 & (df.Sex == "female")], var = 'Age' , target = 'Survived' , row = 'Sex' )
plt.show()

plot_distribution(df[(df.Survived) == 1 & (df.Sex == "Male")], var = 'Age' , target = 'Survived' , row = 'Sex' )
plt.show()

plot_distribution(df[(df.Survived) == 0 & (df.Sex == "Male")], var = 'Age' , target = 'Survived' , row = 'Sex' )
plt.show()

plot_distribution(df, var = "Age", target = "Survived", row = "Sex")
plt.show()

sns.barplot(x="SibSp", y="Survived", hue="Sex", data=df)
#plt.show()

sns.barplot(x="Parch", y="Survived", hue="Sex", data=df)
#plt.show()

sns.barplot(x="SibSp", y="Survived", hue="Sex", data=df)
#plt.show()

sns.barplot(x="Embarked", y="Survived", hue="Sex", data=df);
#plt.show()

X = df

y = X.pop("Survived")
#
#-------------------- CODE FAILS DURING DATA MODELING -------------------------#
#
#model = DecisionTreeClassifier(criterion="gini")
#
#model.fit(X, y)
#model.score(X, y)

#predicted = model.predict(test_data)

#print(predicted)
