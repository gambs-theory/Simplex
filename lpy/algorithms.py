from .data import Mat

class STATUS():
  OPTIMAL = 0
  INFEASIBLE = 1
  UNBOUNDED = 2

class Simplex():
  @staticmethod
  def solve (tableau, minimize=True):
    # Handle feasible basic solution (FBS)
    # TODO

    # tableau.base = search_fbs()

    # Assuming x = 0 is a FBS (i.e putting the slack variable in the base)
    tableau.base = tableau.labels[-tableau.mat.m + 1:]

    while True:
      print (tableau)

      # Default: Minimization Problem
      # Find the most negative reduced cost
      # best, index = tableau.get_row_min(0)
      best, index = min(zip(tableau.mat[0, :], range(len(tableau.mat[0]))))

      # Stop criteria
      if best >= 0:
        print (tableau.base, end=' = ')
        print (tableau.mat[1:, tableau.mat.n - 1])
        return tableau.base

      in_var = tableau.labels[index]
      print (f"{in_var} enters the base; ", end='')

      # Find the maximum step
      # xB = tableau.get_column(tableau.n - 1)
      xB = tableau.mat[:, tableau.mat.n - 1]
      # d  = tableau.get_column(index)
      d = tableau.mat[:, index]

      max_step = float('inf')
      pivot = -1

      # xBi / di
      for i in range(1, len(xB)):
        if d[i] == 0:
          continue # Ignore inf
        
        theta = xB[i]/ d[i]
        if theta < 0:
          continue # Ignora negative steps
      
        if theta < max_step:
          max_step = theta
          pivot = i

      # Pivoting
      if not pivot == -1:
        out_var = tableau.base[pivot - 1]
        print (f"{out_var} leaves the base\n")

        # Update the base
        tableau.base[pivot - 1] = in_var

        # Create the Q matrix
        key_value = d[pivot]
        #print (f"Key value = {key_value}")
        
        # Implement a method to create a identity matrix
        Q = Mat.I(tableau.mat.m)
        Q[pivot, pivot] /= key_value
        
        tableau.mat = Q * tableau.mat

        for i in range(len(d)):
          if not i == pivot:
            k = d[i]
            Q = Mat.I(tableau.mat.m)
            Q[i, pivot] = -k
            #print (f"\nQ{i + 2}:\n{Q}\n")
            tableau.mat = Q * tableau.mat

      # Repeat the process
      # print (tableau)    
      else: 
        print ("Unbounded Solution")
        return 