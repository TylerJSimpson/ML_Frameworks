def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities) 
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    # You need to return this variable correctly
    total_cost = 0
    
    ### START CODE HERE ###  
    # Variable to keep track of sum of cost from each example
    cost_sum = 0

    # Loop over training examples
    for i in range(m):
        # Your code here to get the prediction f_wb for the ith example
        f_wb = w * x[i] + b
        # Your code here to get the cost associated with the ith example
        cost = (f_wb - y[i]) ** 2 
        # Add to sum of cost for each example
        cost_sum = cost_sum + cost 

     # Get the total cost as the sum divided by (2*m)
    total_cost = (1 / (2 * m)) * cost_sum
     ### END CODE HERE ### 

    return total_cost
