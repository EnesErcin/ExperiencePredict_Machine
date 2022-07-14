# ExperiencePredict_Machine
This repository is an example of how logistic regression model can be implemented to make a prediction.

## About the project
Since it is my first attempt to practice machine learning, I tried to avoid using libaries which are built specificly for that purpose such as skitlearn,pytorch and tensor flow. The methodology is inspired from the first required coding assigment from **Andre Ng's deep learning course** which uses the * *images as data* * and predict if the image is cat or not. In order to understand fundamental concepts of artificial nerual network more deeply and discover potential hardships along the way I tired to build a similar model for different type of data.

The data is about people who have occupations related to digital data such as data scientists, data analysts, computer vision engineer and machine learning engineer. Information such as their * *salary, language, countary they work in, remote working ratio and the size of their company* * is given. The objective is to determine their (which is also given) experience level with using the other information. 

Ironically only after I build and run the model the big picture become more obvious to me.

### How
Data gathered was initally a single csv file that is why the first quest is to convert the data in approprate form.

1-Data cleaning and converting
Unusseful data columns are filtered out with **pandas libary**.
Data columns which contain strings are converted to numbers with **skitlearn preprocess function** then each value is divided into largest value in order to appropraitly use * *sigmoid function* * both in form of **numpy arrays**.

The dimension of the weigth matrix is determined as 2 and which are number of independent varibles and 1. I am aware the it is the most basic possible weigth function that is why my next model will focus on that. 

2- Building the model
Builded a propagate functiont that gives an output of dradients and costs.
Adjust the weigths with optimize function for a given learning rate. 

### Results 
The model have different level of confidence for different experty levels. 
For EX the model can predict with %95 accuracy while for MI the model can predcit with %65 accuracy.
To visuallise the learning rate the cost function is printed for each itterations.
Since the data is not large enough for the model learning rate slows down drastically after a while.

