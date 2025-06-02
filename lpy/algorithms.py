from .data import Mat

class STATUS():
  OPTIMAL = 0
  INFEASIBLE = 1
  UNBOUNDED = 2

class Simplex():
  @staticmethod
  def phase1(model):
    from .model import Variable, Expression, Model, TYPE

    print ("PHASE 1 " + 80 * "=")
    
    # Declare your artificial problem
    summation = Expression(None)

    # Checking the relation between the slack and resource vector
    for index, constr in enumerate(model.constrs.values()):
      slack = constr.expr.get_slack()
      if slack == None:
        # Equality constraint - not result of the standard form
        # Add artificial variable if x = 0 doesn't satisfy this constraint
        if not constr.resource.coef == 0:
          summation += Variable (name=f"a{index}", type=TYPE.ARTIFICIAL)
      else:
        # Verify non-negativity satisfation
        if slack.coef/ constr.resource.coef < 0:
          # Non-negativity constraint broken! Add artificial variable
          summation += Variable (name=f"a{index}", type=TYPE.ARTIFICIAL)

    if len (summation.terms) == 0:
      # return 0, dict((slack, 0) for slack in model.get_slack())
      return 0, model.get_slack()
    
    # Solve the artifical problem
    else:
      # Create the artifical problem
      artificial = Model()
      for var in model.vars.values():
        artificial.add_var(var)
      for term in summation.terms:
        artificial.add_var(term.var)

      # The initial basic feasible solution to the artificial problem
      bfs = list()
      for index, constr in enumerate(model.constrs.values()):
        cpy = constr.copy (constr)
        artificial_term = summation[f'a{index}']
        if not artificial_term == None:
          cpy.expr += artificial_term
          bfs.append (artificial_term.var)
        else:
          bfs.append (constr.expr.get_slack().var)
        artificial.add_constr(cpy)

      artificial.set_objective(summation)

      return Simplex.phase2 (artificial, bfs)

  @staticmethod
  def phase2(model, bfs):
    print ("PHASE 2 " + 80 * "=")
    from .model import Tableau, TYPE
    tableau = Tableau (model)
    print (model)
    tableau.update(bfs)
    print (tableau)

    while True:
    #for i in range(2):
      # Default: Minimization Problem
      # Find the most negative reduced cost and take the value and it variable
      best, index = min(zip(tableau.mat[0, :], range (len(tableau.mat[0]) - 1)))

      # Stop criteria

      # Avoid float point imprecision
      if round(best, 6) >= 0:
        fitness = tableau.mat[0, tableau.mat.n - 1]

        for var, value in zip (bfs, tableau.mat[1:, tableau.mat.n - 1]):
          var.set_value (value)

        return fitness, bfs

      in_var = list(tableau.labels)[index]

      # in_var = tableau.labels[index]
      print (f"{in_var} enters the base; ", end='')

      # Find the maximum step
      xB = tableau.mat[:, tableau.mat.n - 1]
      d = tableau.mat[:, tableau.labels[in_var]]

      min_step = float('inf')
      pivot = -1

      for i in range(1, len(xB)):
        if d[i] <= 0:
          continue # Ignore inf
        
        theta = xB[i]/ d[i]
        if theta < 0:
          continue # Ignora negative steps
      
        if theta < min_step:
          min_step = theta
          pivot = i

      # Pivoting
      if not pivot == -1:
        out_var = bfs[pivot - 1]
        print (f"{out_var} leaves the base!")
        bfs[pivot - 1] = in_var
        print (f"New base: {bfs}")
        tableau.update(bfs)
        print (tableau)

      # Repeat the process
      # print (tableau)    
      else: 
        print ("\nUnbounded Solution")
        return 0, None