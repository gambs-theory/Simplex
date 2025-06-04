from .data import Mat

class STATUS():
  OPTIMAL = 0
  INFEASIBLE = 1
  UNBOUNDED = 2

class Simplex():
  @staticmethod
  def phase1(model):
    # Unconstrained problem is unbounded

    from .model import Variable, Expression, Model, Term, TYPE

    # print ("PHASE 1 " + 80 * "=")
    # print (model.vars)
    # print (model)
    # print (80 * "=")
    
    # Associate the constraint to it artificial or slack variable
    association = dict()
    artificial_count = 0

    # Checking the relation between the slack and resource vector
    for index, constr in enumerate(model.constrs):
      slack = constr.expr.get_slack()
      if slack == None:
        # Equality constraint - not result of the standard form
        # Add artificial variable if x = 0 doesn't satisfy this constraint
        if not constr.resource.coef == 0:
          association[f"{index}"] = Term(1, Variable(name=f"a{index}", type=TYPE.ARTIFICIAL))
          artificial_count += 1
      
      # Verify non-negativity satisfation
      else:
        if not constr.resource.coef == 0: 
          # if slack.coef/ constr.resource.coef < 0:
          if slack.coef < 0:
            association[f"{index}"] = Term(1, Variable(name=f"a{index}", type=TYPE.ARTIFICIAL))
            artificial_count += 1
          else:
            association[f"{index}"] = slack
        else:
          association[f"{index}"] = slack

    # if len (summation.terms) == 0:
    if artificial_count == 0:
      # return 0, dict((slack, 0) for slack in model.get_slack())
      base = list()
      for term in association.values():
        base.append(term.var)

      return 0, base
    
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
      base = list()
      for index, constr in enumerate(model.constrs):
        cpy = constr.copy (constr)
        term = association[f"{index}"]

        if term.var.type == TYPE.ARTIFICIAL:
          cpy.expr += term
          base.append(term.var)
        else:
          base.append(term.var)

        artificial.add_constr(cpy)

      artificial_objective = Expression(None)
      for term in association.values():
        if term.var.type == TYPE.ARTIFICIAL:
          artificial_objective += term

      artificial.set_objective(artificial_objective)
      return Simplex.phase2 (artificial, base)

  @staticmethod
  def phase2(model, base):
    print ("PHASE 2 " + 80 * "=")
    from .model import Tableau, TYPE
    tableau = Tableau (model)
    tableau.pivot_base(base)
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

        for var, value in zip (base, tableau.mat[1:, tableau.mat.n - 1]):
          var.set_value (value)

        return fitness, base

      in_var = list(tableau.labels)[index]

      print (f"{in_var} enters the base; ", end='')

      xB = tableau.mat[1:, tableau.mat.n - 1]
      d = tableau.mat[1:, index]

      print (len(xB))
      min_step = float('inf')
      pivot = -1

      for i in range(0, len(xB)):
        if d[i] <= 0:
          continue 
        
        theta = xB[i]/ d[i]
        if theta < 0:
          continue
        
        if theta < min_step:
          min_step = theta
          pivot = i

        # Priorize artificial variable
        elif theta == min_step and base[i].type == TYPE.ARTIFICIAL:
          pivot = i

      # Pivoting
      if not pivot == -1:
        out_var = base[pivot]
        print (f"{out_var} leaves the base!", end='; ')
        base[pivot] = in_var
        print (f"New base: {base}")
        tableau.pivot_base(base)
        print ("\n", tableau)
      else:
        return 0, None