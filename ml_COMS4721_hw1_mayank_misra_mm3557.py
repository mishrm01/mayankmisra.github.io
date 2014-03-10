# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Name:  Mayank Misra | Uni: mm3557 | ml_COMS4721_hw1_mayank_misra_mm3557

# <markdowncell>

# #Preliminaries

# <markdowncell>

# ### Question 1: True or False?
# #### 1. Supervised and unsupervised learning both aim to identify classes in data (at the learning stage not prediction).
# ####    Answer>>> True
# #### 2. When feature space is large, overfitting is likely.
# ####    Answer>>> True
# #### 3. Overfitting can be controlled by regularization.
# ####    Answer>>> True
# #### 4. Once you learn a classification model, you can use the test set to assess the model performance.
# ####    Answer>>> True
# #### 5. If the performance of a classification model on the test set is poor, you can re-calibrate your model parameters to achieve a better model.
# ####    Answer>>> False
# #### 6. Cross-validation is used only when one have a large training set.
# ####    Answer>>> False
# #### 7. The examples in a validation set are used to train a classification model.
# ####    Answer>>> False
# #### 8. To learn a regression model you can either use gradient descent or normal equations.
# ####    Answer>>> True
# #### 9. Because it is straightforward to calculate in just one step, Normal equation is the preferred method when the feature space is large (e.g., 10,000 features).
# ####    Answer>>> False
# #### 10. If the learning rate α is small enough, gradient descent converges very fast.
# ####    Answer>>> False
# #### 11. Ridge regression aims to increase the variance of linear regression by decreasing the bias.
# ####    Answer>>> False
# #### 12. Lasso is a variant of linear regression that calculates a sparse solution.
# ####    Answer>>> True
# #### 13. K-NN works only for classification.
# ####    Answer>>> False
# #### 14. K-NN is a linear classification method.
# ####    Answer>>> False
# #### 15. The loss function aggregates the classification/regression error on all examples.
# ####    Answer>>> True

# <markdowncell>

# ### Question 2: Machine Learning Definition in Practice
# #### What are the sets E, T, and P in the case of a Recommender System? Please justify your answer and elaborate with examples.
# 
# #### Recall: “A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E. ” Tom Mitchell.
# 
# ####   Answer>>> Recommendation systems are a subclass of information filtering system that seek to predict the 'rating' or 'preference' that user would give to an item. (Francesco Ricci and Lior Rokach and Bracha Shapira, Introduction to Recommender Systems Handbook, Recommender Systems Handbook, Springer, 2011, pp. 1-35.  http://www.inf.unibz.it/~ricci/papers/intro-rec-sys-handbook.pdf) 
# 
# ####   In a recommendation system, the available data on the user (user matrix of purchase history, likes/dislikes, ratings, surveys, profile) is mined to predict future behavior (buying pattern, product preferences) in attempt to provide a contextual and personalized experience.  
# 
# ####   Experience E:  
# ####   The trainging Experience is the learning simulations where predictions from the model are matched with known results. For example, a set of movie likes/dislikes can be used to model the probability of the user liking a particular movie. These experiences are used to further refine the model to make it better (it learns form its experiences).  
# 
# ####   Tasks T:  
# ####   The act of predicting/recommending a specific user behavior is the task.   
# 
# ####   Performance P:  
# ####   Performance of a recommendation systemis the accuracy with which the underlying model is able to predict the actual behaviour of a user, given the training datset (of user preferences and past behaviors).  This result will feedback into the user matrix and will become part of future predictions therby improving the acurracy of the model.   The training dataset is the past data on users that captures categorical and descriptive information (like/dislike, age, purchases etc).

# <markdowncell>

# # Practical Problems

# <markdowncell>

# ## 1. Linear regression with one feature
# #### We are interested in studying the relationship between age and height (statures) in girls aged 2 to 8 years old. We think that this can be modeled with a linear regression model. We have examples (data points) for a population of 60 girls. Each example has one feature Age along with a numerical label Height. We will use the dataset girls_train.csv (derived from CDC growthchart data1). Your mission is to implement linear regression with one feature using gradient descent. You will plot the data, regression line, coefficient contours and cost function. You will finally make a prediction for a new example using your regression model and assess your out of sample mean square error.

# <markdowncell>

# ### 1.1 Load & Plot
# #### a) Load the dataset girls.csv.
# #### b) Plot the distribution of the data. You should be getting a plot similar to Figure 1.

# <codecell>

import numpy as np
# include numpy libraries to facilitate reading text files like csv
import matplotlib as mpl
# include the matplot libraries
import matplotlib.pyplot as plt
# include plotting libraries

def getData():
    # specify the data file that will be read
    age, height = np.loadtxt('/Users/mayank/Dropbox/DataSets/ColumbiaUniversity/ML/hw1_all/girls_train.csv', 
                             dtype='float', 
                             comments='# The file contains a null third row.  Force loadtxt to read first two columns', 
                             delimiter=',', 
                             converters=None, 
                             skiprows=0, 
                             usecols=(0,1), 
                             unpack=True, 
                             ndmin=0)
    print (age)
    print (height)
    
    fig = plt.figure()
    axl = fig.add_subplot(1,1,1,axisbg='white')
    plt.plot(age, height, 'ro')
    plt.title('Age and Stature')
    plt.xlabel('Age in Years')
    plt.ylabel('Height in Meters')
    plt.show()

getData()


# <markdowncell>

# ####The plot of girls_train.csv looks like this:  [Age and Stature](https://pbs.twimg.com/media/BiUfDq5IgAEgENU.png)

# <markdowncell>

# ### 1.2 Gradient descent
# #### Now that the data is loaded and you know how it looks like, find a regression model of the form: Height=β0 +β1 ×Age
# #### a) Implement Gradient Descent to find the β’s of the model. Remember you need to add the vector 1 ahead of your data matrix. Also, make sure you update the parameters β’s simultaneously. Use a learning rate alpha = 0.05 and #iterations = 1500.
# #### b) What is the mean square error of your regression model on the training set?

# <codecell>

import numpy as np
# include numpy libraries to facilitate reading text files like csv
import matplotlib as mpl
# include the matplot libraries
import matplotlib.pyplot as plt
# include plotting libraries

# read the data
def getData():
    # specify the data file that will be read
    x, y = np.loadtxt('/Users/mayank/Dropbox/DataSets/ColumbiaUniversity/ML/hw1_all/girls_train.csv', 
                             dtype='float', 
                             comments='# The file contains a null third row.  Force loadtxt to read first two columns', 
                             delimiter=',', 
                             #converters=None, 
                             #skiprows=0, 
                             usecols=(0,1), 
                             unpack=True, 
                             ndmin=0
                             )
    return x, y

#initialize descent parameters
iteratNum = 1500
alpha = 0.05

#calculate Gradient Descent for 'n' examples with 'm' features and learning rate 'alpha' and iteration number 'iteatNum'
# m and n don't need to be passed explicitly as numpy will calculate it itself

def gradientDescent(x, y, beta, alpha, m, iteratNum):
    xTranspose = x.transpose()
    for i in range(0, iteratNum):
        #hypothesis h = X * beta
        hypothesis = np.dot(x, beta)
        
        #loss = hypothesis - y
        loss = hypothesis - y
        
        #cost is the average cost per row, squared cost (loss^2)/2m where m is the number of examples
        cost = np.sum(loss ** 2) / (2 * m)
        
        print("Iteration %d | Cost: %f" % (i, cost))
        
        # avg gradient per example where gradient = X' * loss / m
        gradient = np.dot(xTranspose, loss) / m
        
        # update beta
        beta = beta - (alpha * gradient)
    
    return beta
# initialize beta and iterate
beta = np.ones(n)
beta = gradientDescent(x, y, beta, alpha, m, iteratNum)
print(beta)


# mean square error


#Display GD result
print "Theta computed from gradient descent:\n",beta

# <markdowncell>

# ### 1.3 Plot the regression line, contours and bowl function

# <codecell>

import numpy as np
# include numpy libraries to facilitate reading text files like csv
import matplotlib as mpl
# include the matplot libraries
import matplotlib.pyplot as plt
# include plotting libraries

def getData():
    # specify the data file that will be read
    age, height = np.loadtxt('/Users/mayank/Dropbox/DataSets/ColumbiaUniversity/ML/hw1_all/girls_train.csv', 
                             dtype='float', 
                             comments='# The file contains a null third row.  Force loadtxt to read first two columns', 
                             delimiter=',', 
                             converters=None, 
                             skiprows=0, 
                             usecols=(0,1), 
                             unpack=True, 
                             ndmin=0)
    #print (age)
    #print (height)
    
    fig = plt.figure()
    axl = fig.add_subplot(1,1,1,axisbg='white')
    #add regression line plot once code resolves correctly
    plt.plot(age, height, 'ro')
    plt.title('Age and Stature')
    plt.xlabel('Age in Years')
    plt.ylabel('Height in Meters')
    plt.show()

getData()

def gradientDescent(x, y, beta, alpha, m, iteratNum):
    xTranspose = x.transpose()
    for i in range(0, iteratNum):
        #hypothesis h = X * beta
        hypothesis = np.dot(x, beta)
        
        #loss = hypothesis - y
        loss = hypothesis - y
        
        #cost is the average cost per row, squared cost (loss^2)/2m where m is the number of examples
        cost = np.sum(loss ** 2) / (2 * m)
        
        # avg gradient per example where gradient = X' * loss / m
        gradient = np.dot(xTranspose, loss) / m
        
        # update beta
        beta = beta - (alpha * gradient)
    
    return beta
# initialize beta and iterate
beta = np.ones(n)
beta = gradientDescent(x, y, beta, alpha, m, iteratNum)
print(beta)

# <markdowncell>

# ### 1.4 Testing your model and Making a prediction for a new example

# <codecell>


# <markdowncell>

# ## 2. Linear regression with multiple features

# <markdowncell>

# #### In this problem, you will work on linear regression with multiple features using gradient descent and the normal equation. You will also study the relationship between the risk function, the convergence of gradient descent, and the learning rate. We will use the dataset girls age weight height 2 8.csv (derived from CDC growthchart data).

# <markdowncell>

# #### 2.1 Data Preparation & Normalization

# <codecell>


