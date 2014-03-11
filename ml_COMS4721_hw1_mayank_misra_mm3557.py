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

# include numpy libraries to facilitate reading text files like csv
import numpy as np
from numpy import loadtxt, zeros, ones, array, linspace, logspace
# include the matplot libraries
import matplotlib as mpl
# include plotting libraries
import matplotlib.pyplot as plt
from pylab import scatter, show, title, xlabel, ylabel, plot, contour

#Load the dataset
data = loadtxt('/Users/mayank/Dropbox/DataSets/ColumbiaUniversity/ML/hw1_all/girls_train.csv',
                             dtype='float', 
                             comments='# The file contains a null third row.  Force loadtxt to read first two columns', 
                             delimiter=',', 
                             #converters=None, 
                             #skiprows=0, 
                             usecols=(0,1), 
                             unpack=True, 
                             ndmin=0
                             ) 
 
#Initialize useful parameters for regression 
def getCost(X, y, beta):

    #Number of training samples
    m = y.size
 
    #calculate the cost of a particular choice of beta
    predictedValue = X.dot(beta).flatten()
 
    #calculate squared error
    sqErrors = (predictedValue - y) ** 2
    
    J = (1.0 / (2 * m)) * sqErrors.sum()
    
    return J
 
#Compute gradient descent (GD) to infer beta. GD steps = numberIterations  and learning rate = alpha
def gradientDescent(X, y, beta, alpha, numberIterations):
    # number of training examples
    m = y.size
    J_history = zeros(shape=(numberIterations, 1))
 
    for i in range(numberIterations):
 
        predictedValue = X.dot(beta).flatten()
 
        sumErrorsCol1 = (predictedValue - y) * X[:, 0]
        sumErrorsCol2 = (predictedValue - y) * X[:, 1]
 
        beta[0][0] = beta[0][0] - alpha * (1.0 / m) * sumErrorsCol1.sum()
        beta[1][0] = beta[1][0] - alpha * (1.0 / m) * sumErrorsCol2.sum()
 
        J_history[i, 0] = getCost(X, y, beta)
 
    return beta, J_history
 
X = data[:, 0]
y = data[:, 1]
 
 
#number of training samples
m = y.size
 
#Add a column of ones to X 
X1 = ones(shape=(m, 2))
X1[:, 1] = X
 
#Initialize beta parameters
beta = zeros(shape=(2, 1))
 
#initialize gradient descent parameters
numberIterations = 1500
alpha = 0.05
 
#compute and display initial cost
print 'Initial cost is %f' % getCost(X1, y, beta)
 
beta, J_history = gradientDescent(X1, y, beta, alpha, numberIterations)
print beta


# <markdowncell>

# ### 1.3 Plot the regression line, contours and bowl function

# <codecell>

# include numpy libraries to facilitate reading text files like csv
import numpy as np
from numpy import loadtxt, zeros, ones, array, linspace, logspace
# include the matplot libraries
import matplotlib as mpl
# include plotting libraries
import matplotlib.pyplot as plt
from pylab import scatter, show, title, xlabel, ylabel, plot, contour

#Load the dataset
data = loadtxt('/Users/mayank/Dropbox/DataSets/ColumbiaUniversity/ML/hw1_all/girls_train.csv',
                             dtype='float', 
                             comments='# The file contains a null third row.  Force loadtxt to read first two columns', 
                             delimiter=',', 
                             #converters=None, 
                             #skiprows=0, 
                             usecols=(0,1), 
                             unpack=True, 
                             ndmin=0
                             ) 
 
#Initialize useful parameters for regression 
def getCost(X, y, beta):

    #Number of training samples
    m = y.size
 
    #calculate the cost of a particular choice of beta
    predictedValue = X.dot(beta).flatten()
 
    #calculate squared error
    sqErrors = (predictedValue - y) ** 2
    
    J = (1.0 / (2 * m)) * sqErrors.sum()
 
    return J
 
#Compute gradient descent (GD) to infer beta. GD steps = numberIterations  and learning rate = alpha
def gradientDescent(X, y, beta, alpha, numberIterations):
    # number of training examples
    m = y.size
    J_history = zeros(shape=(numberIterations, 1))
 
    for i in range(numberIterations):
 
        predictedValue = X.dot(beta).flatten()
 
        sumErrorsCol1 = (predictedValue - y) * X[:, 0]
        sumErrorsCol2 = (predictedValue - y) * X[:, 1]
 
        beta[0][0] = beta[0][0] - alpha * (1.0 / m) * sumErrorsCol1.sum()
        beta[1][0] = beta[1][0] - alpha * (1.0 / m) * sumErrorsCol2.sum()
 
        J_history[i, 0] = getCost(X, y, beta)
 
    return beta, J_history
 
X = data[:, 0]
y = data[:, 1]
 
 
#number of training samples
m = y.size
 
#Add a column of ones to X 
X1 = ones(shape=(m, 2))
X1[:, 1] = X
 
#Initialize beta parameters
beta = zeros(shape=(2, 1))
 
#initialize gradient descent parameters
numberIterations = 1500
alpha = 0.05
 
#compute beta
 
beta, J_history = gradientDescent(X1, y, beta, alpha, numberIterations)



## PLOTS 

#Plot the data
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
    fig = plt.figure()
    axl = fig.add_subplot(1,1,1,axisbg='white')
    plt.plot(age, height, 'ro')
    plt.title('Age and Stature')
    plt.xlabel('Age in Years')
    plt.ylabel('Height in Meters')
    plt.show()

getData()

#Plot the results
result = X1.dot(beta).flatten()
plot(data[:, 0], result)
show()
 
#plot values of beta
#initialize matrix space for beta and cost values.  
beta0 = linspace(-10, 10, 100)
beta1 = linspace(-1, 4, 100)
J_beta0beta1 = zeros(shape=(beta0.size, beta1.size))
#Fill the matrix J_beta0beta1
for t1, element in enumerate(beta0):
    for t2, element2 in enumerate(beta1):
        iBeta = zeros(shape=(2, 1))
        iBeta[0][0] = element
        iBeta[1][0] = element2
        J_beta0beta1[t1, t2] = getCost(X1, y, iBeta)
 
#graph a contour plot
J_beta0beta1 = J_beta0beta1.T
contour(beta0, beta1, J_beta0beta1, logspace(-2, 3, 20))
xlabel('beta_0')
ylabel('beta_1')
scatter(beta[0][0], beta[1][0])
show()

# <markdowncell>

# ### 1.4 Testing your model and Making a prediction for a new example

# <codecell>

# include numpy libraries to facilitate reading text files like csv
import numpy as np
from numpy import loadtxt, zeros, ones, array, linspace, logspace
# include the matplot libraries
import matplotlib as mpl
# include plotting libraries
import matplotlib.pyplot as plt
from pylab import scatter, show, title, xlabel, ylabel, plot, contour

#Load the dataset
data = loadtxt('/Users/mayank/Dropbox/DataSets/ColumbiaUniversity/ML/hw1_all/girls_train.csv',
                             dtype='float', 
                             comments='# The file contains a null third row.  Force loadtxt to read first two columns', 
                             delimiter=',', 
                             #converters=None, 
                             #skiprows=0, 
                             usecols=(0,1), 
                             unpack=True, 
                             ndmin=0
                             ) 
 
#Initialize useful parameters for regression 
def getCost(X, y, beta):

    #Number of training samples
    m = y.size
 
    #calculate the cost of a particular choice of beta
    predictedValue = X.dot(beta).flatten()
 
    #calculate squared error
    sqErrors = (predictedValue - y) ** 2
    
    J = (1.0 / (2 * m)) * sqErrors.sum()
 
    return J
 
#Compute gradient descent (GD) to infer beta. GD steps = numberIterations  and learning rate = alpha
def gradientDescent(X, y, beta, alpha, numberIterations):
    # number of training examples
    m = y.size
    J_history = zeros(shape=(numberIterations, 1))
 
    for i in range(numberIterations):
 
        predictedValue = X.dot(beta).flatten()
 
        sumErrorsCol1 = (predictedValue - y) * X[:, 0]
        sumErrorsCol2 = (predictedValue - y) * X[:, 1]
 
        beta[0][0] = beta[0][0] - alpha * (1.0 / m) * sumErrorsCol1.sum()
        beta[1][0] = beta[1][0] - alpha * (1.0 / m) * sumErrorsCol2.sum()
 
        J_history[i, 0] = getCost(X, y, beta)
 
    return beta, J_history
 
X = data[:, 0]
y = data[:, 1]
 
 
#number of training samples
m = y.size
 
#Add a column of ones to X 
X1 = ones(shape=(m, 2))
X1[:, 1] = X
 
#Initialize beta parameters
beta = zeros(shape=(2, 1))
 
#initialize gradient descent parameters
numberIterations = 1500
alpha = 0.05


#Predict height for girl aged 4.5

predict1 = array([4.5, 1]).dot(beta).flatten()
print 'For a girl of age = 4.5, the prediction for height is %f' % (predict1)

# <markdowncell>

# ## 2. Linear regression with multiple features

# <markdowncell>

# #### In this problem, you will work on linear regression with multiple features using gradient descent and the normal equation. You will also study the relationship between the risk function, the convergence of gradient descent, and the learning rate. We will use the dataset girls age weight height 2 8.csv (derived from CDC growthchart data).

# <markdowncell>

# #### 2.1 Data Preparation & Normalization

# <codecell>


