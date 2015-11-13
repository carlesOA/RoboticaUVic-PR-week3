import numpy as np
from scipy.linalg import lstsq
dot = np.dot	
inv = np.linalg.inv 

print "Solving Q1 and Q2\n"

data=np.loadtxt('housing.data')  #load the imput data

y = data[:,-1] 	#get the last colum (-1) of the imput data

avg = np.mean (y) #do the average of y

l = len(y)  #the length of y

X = np.ones ([l,1]) #creat a vector of l colums with 1 Bias

theta = dot(dot(inv(dot(X.T, X)), X.T), y) #obtain theta
print "The value of theta is:", theta

MSE = sum((y-np.dot(X, theta))**2) / l #obtain MSE
print "The value of MSE is:", MSE

print "Solving Q3 \n"

me = l/2		# Split the data in two parts (50%-50%)
training = y[0:me]
testing = y[me:l]


for n in range(0,len(data[0])-1):
    Q = np.hstack((X[0:me], data[0:me,n].reshape(len(training),1)))
    theta = lstsq(Q,training)[0]
    MSE = sum((training-np.dot(Q, theta))**2) / len(training)
        
    if n == 0:		#look for the most informative
	MSE_i = MSE
    if MSE_i > MSE:
        MSE_i = MSE
        num = n


for n in range(0,len(data[0])-1):
    Q = np.hstack((X[0:me], data[0:me,n].reshape(len(training),1)))
    theta = lstsq(Q,training)[0]
    MSE = sum((testing-np.dot(Q, theta))**2) / len(testing)
           
    if n == 0:
	MSE_m = MSE
	MSE_p = MSE
    if MSE_m > MSE:	#look for the better
        MSE_m = MSE
        num_m = n
    if MSE_p < MSE:	#look for the worst
        MSE_p = MSE
        num_p = n



mean = sum(testing) / len(testing)	#look for the mean

VAR = sum((mean-testing)**2) / len(testing)	#look for the variance

FVU = MSE_m/VAR	#look for the coefficient of determination

R = 1 - FVU

print "The coefficient of determination for the best is:", R

FVU = MSE_p/VAR

R = 1 - FVU

print "The coefficient of determination for the worst is:", R






