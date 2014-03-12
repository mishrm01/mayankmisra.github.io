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
# #### Load the dataset girls.csv. and plot the distribution of the data. 
# ####The plot of girls_train.csv looks like this:  [Age and Stature](https://pbs.twimg.com/media/BiUfDq5IgAEgENU.png)

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
# #### The β’s of the model with a learning rate alpha = 0.05 and #iterations = 1500.
# #### The mean square error of your regression model on the training set?

# <codecell>

#Summary
#Load the dataset - csv with no headers
#Initialize useful parameters for regression
#Compute gradient descent (GD) to infer beta. GD steps = numberIterations  and learning rate = alpha
#initialize gradient descent parameters
#compute and display initial cost
#Predict height for girl aged 4.5
#code ported from http://bit.ly/1iu3Wke (based on Ex1 from Andrew Ng Coursera ml-class.org)
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

# <markdowncell>

# #### [Link to the resulting Contour Plot](https://pbs.twimg.com/media/BibJ88pIgAACG_k.png)

# <codecell>

#Summary
#Load the dataset - csv with no headers
#Initialize useful parameters for regression
#Compute gradient descent (GD) to infer beta. GD steps = numberIterations  and learning rate = alpha
#initialize gradient descent parameters
#compute and display initial cost
#Predict height for girl aged 4.5
#code ported from http://bit.ly/1iu3Wke (based on Ex1 from Andrew Ng Coursera ml-class.org)
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
    #y = -0.02288278 + 1.03139807*x
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
plot(data[:, 1], result)
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

# #### [Contour Plot](https://pbs.twimg.com/media/BibJ88pIgAACG_k.png)

# <markdowncell>

# ### 1.4 Testing your model and Making a prediction for a new example

# <markdowncell>

# #### For a girl of age = 4.5, the prediction for height is 0.928426

# <codecell>

#Summary
#Load the dataset - csv with no headers
#Initialize useful parameters for regression
#Compute gradient descent (GD) to infer beta. GD steps = numberIterations  and learning rate = alpha
#initialize gradient descent parameters
#compute and display initial cost
#Predict height for girl aged 4.5
#code ported from http://bit.ly/1iu3Wke (based on Ex1 from Andrew Ng Coursera ml-class.org)
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

beta, J_history = gradientDescent(X1, y, beta, alpha, numberIterations)


#Predict height for girl aged 4.5

predict1 = array([4.5, 1]).dot(beta).flatten()
print 'For a girl of age = 4.5, the prediction for height is %f' % (predict1)


# <markdowncell>

# ### Compare mean sqaured error of test set with training set

# <markdowncell>

# #### Initial cost for train set:  1.193600
# #### Initial cost for test set:  1.232898
# #### The beta values for train set:  -0.07334643 and 1.03658222
# #### The beta values for test set:  -0.02288278 and 1.03139807

# <codecell>

#Summary
#Load the dataset - csv with no headers
#Initialize useful parameters for regression
#Compute gradient descent (GD) to infer beta. GD steps = numberIterations  and learning rate = alpha
#initialize gradient descent parameters
#compute and display initial cost
#Predict height for girl aged 4.5
#code ported from http://bit.ly/1iu3Wke (based on Ex1 from Andrew Ng Coursera ml-class.org)
# include numpy libraries to facilitate reading text files like csv
import numpy as np
from numpy import loadtxt, zeros, ones, array, linspace, logspace
# include the matplot libraries
import matplotlib as mpl
# include plotting libraries
import matplotlib.pyplot as plt
from pylab import scatter, show, title, xlabel, ylabel, plot, contour

#Load the dataset
data = loadtxt('/Users/mayank/Dropbox/DataSets/ColumbiaUniversity/ML/hw1_all/girls_test.csv',
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

# #### Initial cost for train set:  1.193600
# #### Initial cost for test set:  1.232898
# #### The beta values for train set:  -0.07334643 and 1.03658222
# #### The beta values for test set:  -0.02288278 and 1.03139807

# <markdowncell>

# ## 2. Linear regression with multiple features

# <markdowncell>

# #### In this problem, you will work on linear regression with multiple features using gradient descent and the normal equation. You will also study the relationship between the risk function, the convergence of gradient descent, and the learning rate. We will use the dataset girls age weight height 2 8.csv (derived from CDC growthchart data).

# <markdowncell>

# #### 2.1 Data Preparation & Normalization
# #### 2.2 Multi variable Gradient Descent on normalized data

# <markdowncell>

# #### Plot the data for exploration and size
# 
# #### [Link to the 3d distribution of data](https://github.com/mayankmisra/mayankmisra.github.io/blob/rel-0.2/assets/Age-Weight-Height-ExploreData.png?raw=true)

# <codecell>

import numpy as np
# include numpy libraries to facilitate reading text files like csv
import matplotlib as mpl
# include the matplot libraries
import matplotlib.pyplot as plt
# include plotting libraries

def getData():
    # specify the data file that will be read
    age, weight, height = np.loadtxt('/Users/mayank/Dropbox/DataSets/ColumbiaUniversity/ML/hw1_all/girls_age_weight_height_2_8.csv',
                             dtype='float', 
                             #comments='#', 
                             delimiter=',', 
                             #converters=None, 
                             #skiprows=0, 
                             #usecols=(0,1), 
                             unpack=True, 
                             ndmin=0
                             )
    print (age)
    print (weight)
    print (height)
    
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    plt.plot(age, weight, height, 'ro')
    plt.title('Age Weight Height')
    ax.set_xlabel('Age in Years')
    ax.set_ylabel('Weight in Kilograms')
    ax.set_zlabel('Height in Meters')
    plt.show()

getData()

# <markdowncell>

# #### Data Preparation, Normalization and Gradient Descent

# <codecell>

#Summary
#Load the dataset - csv with no headers
#Initialize useful parameters for regression
#Compute gradient descent (GD) to infer beta. GD steps = numberIterations  and learning rate = alpha
#initialize gradient descent parameters
#compute and display initial cost
#Predict height for girl aged 4.5
#code ported from http://bit.ly/1fnpcVn (based on Andrew Ng Coursera ml-class.org)
# include numpy libraries to facilitate reading text files like csv
from numpy import loadtxt, zeros, ones, array, linspace, logspace, mean, std, arange
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import plot, show, xlabel, ylabel

#Load the dataset
data = loadtxt('/Users/mayank/Dropbox/DataSets/ColumbiaUniversity/ML/hw1_all/girls_age_weight_height_2_8.csv',
                             dtype='float', 
                             #comments='#', 
                             delimiter=',', 
                             #converters=None, 
                             #skiprows=0, 
                             #usecols=(0,1), 
                             unpack=True, 
                             ndmin=0
                             )
                             
#Normalize features- implies regress data so that mean is 0 and standard deviation is 1.  
def normalizeFeatures(X):
    normalizedStdDev = []
    normalizedMean = []
 
    normalizeX = X
 
    rangeX = X.shape[1]
    for i in range(rangeX):
        getStdDev = std(X[:, i])
        getMean = mean(X[:, i])
        normalizedStdDev.append(getStdDev)
        normalizedMean.append(getMean)
        normalizeX[:, i] = (normalizeX[:, i] - getMean) / getStdDev
 
    return normalizeX, normalizedMean, normalizedStdDev
 
#Initialize useful parameters for regression 
def getCost(X, y, beta):
    #calculate training examples
    m = y.size
    #calculate the cost of a particular choice of beta
    predictedValue = X.dot(beta)
    #calculate squared error
    sqErrors = (predictedValue - y)
 
    J = (1.0 / (2 * m)) * sqErrors.T.dot(sqErrors)
 
    return J

#Compute gradient descent (GD) to infer beta. GD steps = numberIterations  and learning rate = alpha 
def gradientDescent(X, y, beta, alpha, numberIterations):
    #number of training examples
    m = y.size
    J_history = zeros(shape=(numberIterations, 1))
 
    for i in range(numberIterations):
        predictedValue = X.dot(beta) 
        betaSize = beta.size
 
        for X1 in range(betaSize):
 
            hold = X[:, X1]
            hold.shape = (m, 1)
            #calculate step errors
            sumErrorsCol = (predictedValue - y) * hold
            #calculate step beta values
            beta[X1][0] = beta[X1][0] - alpha * (1.0 / m) * sumErrorsCol.sum()
 
        J_history[i, 0] = getCost(X, y, beta)
 
    return beta, J_history
 
#Initialize GD regression and print risk/cost
X = data[:, :2]
y = data[:, 2]
 
#number of training samples
m = y.size

y.shape = (m, 1)
 
#Normalize features- implies regress data so that mean is 0 and standard deviation is 1. 
x, normalizedMean, normalizedStdDev = normalizeFeatures(X)
 
#Add a column of ones to X
X1 = ones(shape=(m, 3))
X1[:, 1:3] = x
 
#initialize gradient descent parameters
iterations = 50
alpha = 0.001
 
#Initialize beta parameters
beta = zeros(shape=(3, 1))
 
beta, J_history = gradientDescent(X1, y, beta, alpha, iterations)
print beta, J_history

# <markdowncell>

# #### 2.3 Plotting Risk function for different learning rates

# <markdowncell>

# #### Run gradient descent and plot the Risk function with respect to the number of iterations for different values of α ∈ {0.005, 0.001, 0.05, 0.1, 0.5, 1}

# <codecell>

#Summary
#Load the dataset - csv with no headers
#Initialize useful parameters for regression
#Compute gradient descent (GD) to infer beta. GD steps = numberIterations  and learning rate = alpha
#initialize gradient descent parameters
#compute and display initial cost
#Predict height for girl aged 4.5
#code ported from http://bit.ly/1fnpcVn (based on Andrew Ng Coursera ml-class.org)
# include numpy libraries to facilitate reading text files like csv
from numpy import loadtxt, zeros, ones, array, linspace, logspace, mean, std, arange
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import plot, show, xlabel, ylabel

#Load the dataset
data = loadtxt('/Users/mayank/Dropbox/DataSets/ColumbiaUniversity/ML/hw1_all/girls_age_weight_height_2_8.csv',
                             dtype='float', 
                             #comments='#', 
                             delimiter=',', 
                             #converters=None, 
                             #skiprows=0, 
                             #usecols=(0,1), 
                             unpack=True, 
                             ndmin=0
                             )
                             
#Normalize features- implies regress data so that mean is 0 and standard deviation is 1.
#For the each feature x (a column in the data matrix)xscaled = (x − μ(x))/(stdev(x))   
def normalizeFeatures(X):
    normalizedStdDev = []
    normalizedMean = []
 
    normalizeX = X
 
    rangeX = X.shape[1]
    for i in range(rangeX):
        getStdDev = std(X[:, i])
        getMean = mean(X[:, i])
        normalizedStdDev.append(getStdDev)
        normalizedMean.append(getMean)
        normalizeX[:, i] = (normalizeX[:, i] - getMean) / getStdDev
 
    return normalizeX, normalizedMean, normalizedStdDev
 
#Initialize useful parameters for regression 
def getCost(X, y, beta):
    #calculate training examples
    m = y.size
    #calculate the cost of a particular choice of beta
    predictedValue = X.dot(beta)
    #calculate squared error
    sqErrors = (predictedValue - y)
 
    J = (1.0 / (2 * m)) * sqErrors.T.dot(sqErrors)
 
    return J

#Compute gradient descent (GD) to infer beta. GD steps = numberIterations  and learning rate = alpha 
def gradientDescent(X, y, beta, alpha, numberIterations):
    #number of training examples
    m = y.size
    J_history = zeros(shape=(numberIterations, 1))
 
    for i in range(numberIterations):
        predictedValue = X.dot(beta) 
        betaSize = beta.size
 
        for X1 in range(betaSize):
 
            hold = X[:, X1]
            hold.shape = (m, 1)
            #calculate step errors
            sumErrorsCol = (predictedValue - y) * hold
            #calculate step beta values
            beta[X1][0] = beta[X1][0] - alpha * (1.0 / m) * sumErrorsCol.sum()
 
        J_history[i, 0] = getCost(X, y, beta)
 
    return beta, J_history
 
#Initialize regression and plot 
X = data[:, :2]
y = data[:, 2]
 
#number of training samples
m = y.size

y.shape = (m, 1)
 
#Normalize features- implies regress data so that mean is 0 and standard deviation is 1. 
x, normalizedMean, normalizedStdDev = normalizeFeatures(X)
 
#Add a column of ones to X
X1 = ones(shape=(m, 3))
X1[:, 1:3] = x
 
#initialize gradient descent parameters
iterations = 50
alpha = .05
 
#Initialize beta parameters
beta = zeros(shape=(3, 1))
 
beta, J_history = gradientDescent(X1, y, beta, alpha, iterations)
#print beta, J_history
plot(arange(iterations), J_history)
xlabel('Iterations')
ylabel('Cost Function')
show()

# <markdowncell>

# #### [alpha = .001](https://github.com/mayankmisra/mayankmisra.github.io/blob/rel-0.2/assets/Age-Weight-Height-alpha001.png?raw=true)
# #### [alpha = .005](https://github.com/mayankmisra/mayankmisra.github.io/blob/rel-0.2/assets/Age-Weight-Height-alpha005.png?raw=true)
# #### [The best convergence is for alpha = .05](https://pbs.twimg.com/media/Biefh9UCEAE3SWH.png)
# #### [alpha = .5](https://github.com/mayankmisra/mayankmisra.github.io/blob/rel-0.2/assets/Age-Weight-Height-alphaDot5.png?raw=true)
# #### [alpha = 1](https://github.com/mayankmisra/mayankmisra.github.io/blob/rel-0.2/assets/Age-Weight-Height-alpha1.png?raw=true)

# <markdowncell>

# #### 2.4 Normal equation  β = (XtX)−1XT y
# ##### Compare the β vector you obtained with gradient descent to the one calculated with normal equation. Are they the same? Why?

# <codecell>


# <markdowncell>

# #### 2.5 Prediction

# <markdowncell>

# ##### a) Using both β vectors (the one obtained with gradient descent and the one obtained with normal equations), make a height prediction for a 5-year old girl weighting 20 kilos (don’t forget to scale!).
# ##### b) Do gradient descent and Normal Equation lead to the same height prediction?

# <codecell>

#Summary
#Load the dataset - csv with no headers
#Initialize useful parameters for regression
#Compute gradient descent (GD) to infer beta. GD steps = numberIterations  and learning rate = alpha
#initialize gradient descent parameters
#compute and display initial cost
#Predict height for girl aged 4.5
#code ported from http://bit.ly/1fnpcVn (based on Andrew Ng Coursera ml-class.org)
# include numpy libraries to facilitate reading text files like csv
from numpy import loadtxt, zeros, ones, array, linspace, logspace, mean, std, arange
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import plot, show, xlabel, ylabel

#Load the dataset
data = loadtxt('/Users/mayank/Dropbox/DataSets/ColumbiaUniversity/ML/hw1_all/girls_age_weight_height_2_8.csv',
                             dtype='float', 
                             #comments='#', 
                             delimiter=',', 
                             #converters=None, 
                             #skiprows=0, 
                             #usecols=(0,1), 
                             unpack=True, 
                             ndmin=0
                             )
                             
#Normalize features- implies regress data so that mean is 0 and standard deviation is 1.
#For the each feature x (a column in the data matrix)xscaled = (x − μ(x))/(stdev(x))   
def normalizeFeatures(X):
    normalizedStdDev = []
    normalizedMean = []
 
    normalizeX = X
 
    rangeX = X.shape[1]
    for i in range(rangeX):
        getStdDev = std(X[:, i])
        getMean = mean(X[:, i])
        normalizedStdDev.append(getStdDev)
        normalizedMean.append(getMean)
        normalizeX[:, i] = (normalizeX[:, i] - getMean) / getStdDev
 
    return normalizeX, normalizedMean, normalizedStdDev
 
#Initialize useful parameters for regression 
def getCost(X, y, beta):
    #calculate training examples
    m = y.size
    #calculate the cost of a particular choice of beta
    predictedValue = X.dot(beta)
    #calculate squared error
    sqErrors = (predictedValue - y)
 
    J = (1.0 / (2 * m)) * sqErrors.T.dot(sqErrors)
 
    return J

#Compute gradient descent (GD) to infer beta. GD steps = numberIterations  and learning rate = alpha 
def gradientDescent(X, y, beta, alpha, numberIterations):
    #number of training examples
    m = y.size
    J_history = zeros(shape=(numberIterations, 1))
 
    for i in range(numberIterations):
        predictedValue = X.dot(beta) 
        betaSize = beta.size
 
        for X1 in range(betaSize):
 
            hold = X[:, X1]
            hold.shape = (m, 1)
            #calculate step errors
            sumErrorsCol = (predictedValue - y) * hold
            #calculate step beta values
            beta[X1][0] = beta[X1][0] - alpha * (1.0 / m) * sumErrorsCol.sum()
 
        J_history[i, 0] = getCost(X, y, beta)
 
    return beta, J_history
 
#Initialize regression and plot 
X = data[:, :2]
y = data[:, 2]
 
#number of training samples
m = y.size

y.shape = (m, 1)
 
#Normalize features- implies regress data so that mean is 0 and standard deviation is 1. 
x, normalizedMean, normalizedStdDev = normalizeFeatures(X)
 
#Add a column of ones to X
X1 = ones(shape=(m, 3))
X1[:, 1:3] = x
 
#initialize gradient descent parameters
iterations = 50
alpha = 0.05
 
#Initialize beta parameters
beta = zeros(shape=(3, 1))
 
beta, J_history = gradientDescent(X1, y, beta, alpha, iterations)
 
#Predict height for a 5 yr old girl weighing 20 kgs
height = array([1.0,   ((5 - normalizedMean[0]) / normalizedStdDev[0]), ((20 - normalizedMean[1]) / normalizedStdDev[1])]).dot(beta)
print 'Predicted height of a 5yr girl with weight 20kg: %f' % (height)

# <codecell>


