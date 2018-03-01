import numpy as np
from sympy import symbols, diff, lambdify
import matplotlib.pyplot as plt

y_real_fn = lambda x: 0.4*(x**2) + 0.3*x + 0.9 + np.random.normal() #add white guassian noise
np_array_mappper = lambda X, FN: np.fromiter((FN(xi) for xi in X), X.dtype, len(X))
to_col_vect = lambda X: np.reshape(X, (-1, 1))

#interval is half-open # for verif during training
# what np.reshape does is convert a 1x0 dim array
# i.e. a list with only one row and no cols as standard python arrays
# to as many rows and as many cols we need,
# this is usually good for matrices, note this is needed because by default a 1D array has no col indexing
# we must add this to accurately get the feel of what you need. i.e. a 1xn matrix in strict terms!!!
# for (1, -1) we say get length and replace with -1 so its like (1, -1)
# whenever you define an array now must reshape it for future matrix muplt!!!
# if we say
# 

# these are real values, to be used for testing errors in training
# we don't reall need this cause we have our function!! 
# but we will use it only for plotting
X_real = to_col_vect(np.arange(-10,11,1))
Y_real = to_col_vect(np_array_mappper(X_real, y_real_fn)) #np.fromiter((y_real_fn(xi) for xi in X_real), X_real.dtype, len(X_real))   # for verif during training
print("X_real:\n", X_real, "\nY_real:\n", Y_real)
Y_real_dict = dict(zip(X_real[:,0].tolist(), Y_real[:,0].tolist()))
# this array will be compared to our original training set, there is no Eout_real atm only Ein_train and E_in_real
print("Final real values (const) Y_real_dict:\n", Y_real_dict)

plt.ion()  # turn on interactive mode
fig = plt.figure(dpi=180)
ax = fig.add_subplot(111)
ax.set_xlabel('x ->')
ax.set_ylabel('y ->')
curve = ax.scatter(X_real, Y_real, color='r', marker='x')
curve1,curve2 = None, None
ax.set_xlim(np.min(X_real), np.max(X_real))
ax.set_ylim(np.min(Y_real), np.max(Y_real))
plt.draw()
plt.pause(0.5)


w1, w2, b = np.random.rand(3)
print("[w1; w2; b] = [", w1, ";", w2, ";", b, "]")

margin = 16 # <11 e.g. overfiting >11 e.g. 17 general error increases

# this testing will be used for traing as -4<x<8 and for out sample testing testing -8<x<8
# full array will be created but x<-4 will never be accessed
X_outsample = X_real[:margin,:] #to 5
#Y_test = to_col_vect(np_array_mappper(X_test, y_real_fn))

# the weird thing is less training data generalises well as compared to more training data
# we have compared this from e_in vs e_out, when we have all of lhs and some of rhs training is bad
# whwhn we have little of rhs but mostly ON THE HIGHER values, training is better
# this means we have to normalise our data so that y does not exceed 1 etc 
# this will make the network easier to generalise
X_train = X_real[(margin-1):,:] # to_col_vect(np.arange(5, 10, 1))
Y_train = to_col_vect(np.zeros((X_train.shape[0],1)))


print("X_train:\n", X_train, "\nY_train:\n", Y_train)

#get a hypothesis based on w1, w2 and b
sym_w1, sym_w2, sym_b = symbols('sym_w1 sym_w2 sym_b',real=True)
h = lambda x: sym_w1*(x**2) + sym_w2*(x) + sym_b

lr = 0.0001 # have a learning rate

# cost fn: mse
# formula: J = 1/m * SUM_i (h(x_i)-y(x_i))^2
# the cost function is actually a function of all weights
# so it is J(w1, w2, b) i.e. we are minimising each of the variables,
# our general equation is set, it can be x^2, x^3 or anything but must be a linear combinations of the weights
# also here is where multi-variable calculus comes into play: when we do gradient decent we need to look at all variables
# individually and find their respective gradients to find a local minima over all
# so each update has a SEPARATE UPDATE RULE
# remember we dont train individual inputs but rather the weights!!

# also remember that y(x_i) will just give us A VAL, NOT an expression!! cause in real-life you have no fn for training only val!
# also for h(x_i) should return a FUNCTION of weights and bias where x_i actually form part of the co-eff!

# so e.g. let x_i= 

#get mse of weights
#y(x) = 2*x

#remember h(x) is a higher order fn so even after x its still a fn!!

# avg a col vector
avgX = lambda X: (np.sum(X))/(X.shape[0])

print("X_train_avg: ", avgX(X_train))

# function for each x_i
# because we are adding noise we dont want to sample noise everytime to get y_real_fn
# because it real life we have a static data x,y
# so we will use dict which only sampled values once in beginning
# this will reduce error
j_i_of_w1_w2_b = lambda x_i: (h(x_i) - Y_real_dict[x_i])**2

# take in any col vector and return sym function with x_i substituted
# fullname: X_col_vect_to_sym_func_col_vect
F_of_X = lambda X, FN: to_col_vect(np.array([FN(xi) for xi in X[:,0]]))

# fi in this case will be a WELL DEFINED symbolic function i.e. no variables only symbols!!
# this => dF/db or dF/dw1 etc i.e. partial differentiation of a col vector!!
Diff_F_wrt_sym_x = lambda F, sym_x: to_col_vect(np.array([diff(fi, sym_x) for fi in F[:,0]]))

J = F_of_X(X_train, j_i_of_w1_w2_b) #in sample error
# now J is cool, it is an array of functions
# we will differentiate J wrt to w1, w2 and b
# next we will add all those terms and divide by total samples to get mean w.r.t w1, w2 and w3

#note this is a vector !! so u diff every elem!!
#every elem is a multi variable function!!!
diff_J_wrt_b, diff_J_wrt_w1, diff_J_wrt_w2 = Diff_F_wrt_sym_x(J, sym_b), Diff_F_wrt_sym_x(J, sym_w1), Diff_F_wrt_sym_x(J, sym_w2) 

#J_sum_terms = np_array_mappper(X_train, J_sum_i) 
print("\nJ_list:\n", J, "\n\ndiffJ_wrt_b:\n", diff_J_wrt_b, "\n\ndiffJ_wrt_w1:\n", diff_J_wrt_w1, "\n\ndiffJ_wrt_w2:\n", diff_J_wrt_w2)

#evaluate a col vector of sym_fn with given variables w1,w2,b
#it has to be a func of 3 var
# the symbols used are global
# this is like a nested lambda function
evalF_of_w1_w2_b = lambda F, w1_val, w2_val, b_val: to_col_vect(np.array([lambdify((sym_w1,sym_w2,sym_b), fi, "numpy")(w1_val, w2_val, b_val) for fi in F[:,0]]))

# a list.reduce function
eval_and_get_avg = lambda F, w1_val, w2_val, b_val: avgX(evalF_of_w1_w2_b(F, w1_val, w2_val, b_val))
# testf = lambdify((sym_w1,sym_w2,sym_b), h(2), "numpy")
# print("h(2): ", h(2), "eval fn h(2), w1=1,w2=2,b=3:\n", testf(1,2,3))

print("Eval J_list_(w1,w2,b): ", evalF_of_w1_w2_b(J, w1, w2, b))

# remove previous lines and squares
def remPlot(plotHandle):
    if plotHandle is not None:
        plotHandle.remove()
    return None

eval_F_given_X_w1_w2_b = lambda X, w1, w2, b: evalF_of_w1_w2_b(F_of_X(X, (lambda xi: h(xi))), w1, w2, b)

# improve the costs
epochs = 119
for i in range(epochs):
    w1 -= lr*eval_and_get_avg(diff_J_wrt_w1, w1, w2, b)
    w2 -= lr*eval_and_get_avg(diff_J_wrt_w2, w1, w2, b)
    b -= lr*eval_and_get_avg(diff_J_wrt_b, w1, w2, b)

    # get our vector of hypothesis fn
    # test in tests for same range as training data
    # test out tests for unseen data
    X_insample = X_train

    # evaluate hypothesis fn with given weights
    # now this will be a plotable float array
    H_eval_in, H_eval_out = eval_F_given_X_w1_w2_b(X_insample, w1, w2, b), eval_F_given_X_w1_w2_b(X_outsample, w1, w2, b)
    #print("X_test:\n", X_test, "\nH_eval:\n", H_eval)
    curve1,curve2 = remPlot(curve1), remPlot(curve2)
    curve1, = ax.plot(X_insample, H_eval_in, 'g--')
    curve2, = ax.plot(X_outsample, H_eval_out, 'b:')
    plt.draw()
    plt.pause(0.001)

    print("Iteration: ", i+1, "\t\tAvg Cost ('J_in'): ", eval_and_get_avg(J, w1, w2, b),  "\t\tAvg out sample Cost ('J_out'): ", eval_and_get_avg(F_of_X(X_outsample, j_i_of_w1_w2_b), w1, w2, b))
#print("J_sum_terms:\n", J_sum_terms)
plt.ioff()
plt.show()