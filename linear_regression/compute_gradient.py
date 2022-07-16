def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities) 
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model  
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
     """
    
    # Number of training examples
    m = x.shape[0]
    
    dj_dw = 0
    dj_db = 0
    
    # Loop over examples
    for i in range(m):  
        # prediction f_wb for the ith example
        f_wb = w * x[i] + b
        
        # gradient for w from the ith example 
        dj_dw_i = (f_wb - y[i]) * x[i]
        
        # gradient for b from the ith example 
        dj_db_i = f_wb - y[i]
        
        # Update dj_db : In Python, a += 1  is the same as a = a + 1
        dj_db += dj_db_i
        
        # Update dj_dw
        dj_dw += dj_dw_i
        
    # Divide both dj_dw and dj_db by m
    dj_dw = dj_dw / m
    dj_db = dj_db / m
        
    return dj_dw, dj_db
