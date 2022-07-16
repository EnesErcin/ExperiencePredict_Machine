# ExperiencePredict_Machine
This repository is an example of how logistic regression model can be implemented to make a prediction.

## About the project
Since it is my first attempt to practice machine learning, I tried to avoid using libraries that are built specifically for that purpose such as skitlearn, PyTorch, and tensor flow. The methodology is inspired by the first required coding assignment from **Andre Ng's deep learning course** which uses the * *images as data* * aredicts if the image is a cat or not. To understand the fundamental concepts of the artificial neural network more deeply and discover potential hardships along the way I tried to build a similar model for a different type of data.

The data is about people who have occupations related to digital data such as data scientists, data analysts, computer vision engineers, and machine learning engineers. Information such as their  * *salary, language, countary they work in, remote working ratio and the size of their company* * is given. The objective is to determine their (which is also given) experience level with using the other information.

Ironically only after I build and run the model did the big picture become more obvious to me.

### How
Data gathered was initially a single CSV file which is why the first quest is to convert the data into the appropriate form.

1-Data cleaning and converting 
Unuseful data columns are filtered out with the **pandas libary**.
Data columns that contain strings are converted to numbers with  **skitlearn preprocess function** tthen each value is divided into the largest value in order to appropriately use * *sigmoid function* * both in form of **numpy arrays**.

The dimension of the weight matrix is determined as 2 and which is the number of independent variables and 1. I am aware that it is the most basic possible weight function which is why my next model will focus on that.

2- Building the model 
Built a propagate function that gives an output of gradients and costs. Adjust the weights with an optimized function for a given learning rate.
Results

### Results 
The model has a different level of confidence for different expert levels. For EX the model can predict with %95 accuracies while for MI the model can predict with %65 accuracies. To visualize the learning rate the cost function is printed for each iteration. Since the data is not large enough for the model learning rate slows down drastically after a while.
