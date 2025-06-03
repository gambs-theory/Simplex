from .data import Mat

class STATUS():
  OPTIMAL = 0
  INFEASIBLE = 1
  UNBOUNDED = 2

class Simplex():
  @staticmethod
  def phase1(model):
    from .model import Variable, Expression, Model, Term, TYPE

    print ("PHASE 1 " + 80 * "=")
    print (model)
    
    # Associate the constraint to it artificial or slack variable
    association = dict()
    artificial_count = 0

    # Checking the relation between the slack and resource vector
    for index, constr in enumerate(model.constrs.values()):
      slack = constr.expr.get_slack()
      if slack == None:
        # Equality constraint - not result of the standard form
        # Add artificial variable if x = 0 doesn't satisfy this constraint
        if not constr.resource.coef == 0:
          association[constr.name] = Term(1, Variable(name=f"a{index}", type=TYPE.ARTIFICIAL))
          artificial_count += 1
      
      # Verify non-negativity satisfation
      else:
        if not constr.resource.coef == 0: 
          # if slack.coef/ constr.resource.coef < 0:
          if slack.coef < 0:
            association[constr.name] = Term(1, Variable(name=f"a{index}", type=TYPE.ARTIFICIAL))
            artificial_count += 1
          else:
            association[constr.name] = slack
        else:
          association[constr.name] = slack

    # if len (summation.terms) == 0:
    if artificial_count == 0:
      # return 0, dict((slack, 0) for slack in model.get_slack())
      return 0, association
    
    # Building the Artificial Linear Programming Problem
    else:
      # Create the artifical problem
      artificial = Model()
      for var in model.vars.values():
        artificial.add_var(var)
      for term in association.values():
        if term.var.type == TYPE.ARTIFICIAL:
          artificial.add_var(term.var)

      # The initial basic feasible solution to the artificial problem
      bfs = list()
      for constr in model.constrs.values():
        cpy = constr.copy (constr)
        term = association[constr.name]

        if term.var.type == TYPE.ARTIFICIAL:
          cpy.expr += term
          bfs.append(term.var)
        else:
          bfs.append(term.var)

        artificial.add_constr(cpy)

      artificial_objective = Expression(None)
      for term in association.values():
        if term.var.type == TYPE.ARTIFICIAL:
          artificial_objective += term

      artificial.set_objective(artificial_objective)
      
      return Simplex.phase2 (artificial, bfs)

  @staticmethod
  def phase2(model, bfs):
    print ("PHASE 2 " + 80 * "=")
    from .model import Tableau, TYPE
    tableau = Tableau (model)
    print (model)
    tableau.update(bfs)

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

      print (f"{in_var} enters the base; ", end='')

      xB = tableau.mat[:, tableau.mat.n - 1]
      d = tableau.mat[:, tableau.labels[in_var]]

      min_step = float('inf')
      pivot = -1

      for i in range(1, len(xB)):
        if d[i] <= 0:
          continue 
        
        theta = xB[i]/ d[i]
        if theta < 0:
          continue
        
        if theta < min_step:
          min_step = theta
          pivot = i

        # Priorize artificial variable
        elif theta == min_step and bfs[i - 1].type == TYPE.ARTIFICIAL:
          pivot = i

      # Pivoting
      if not pivot == -1:
        print (f"0* = {min_step}")
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