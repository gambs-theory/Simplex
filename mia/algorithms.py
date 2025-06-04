from .data import Mat

class Simplex():
  @staticmethod
  # Argument: 
  #   model
  # Return:
  # - The phase2 return
  
  def phase1(model):
    # Unconstrained problem is unbounded
    # TODO

    from .model import Variable, Expression, Model, Term, TYPE, STATUS,  Tableau

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

      return Tableau(model, base), STATUS.UNSOLVED
    
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
      
      # Create the tableau of the artificial problem
      t = Tableau(artificial, base)
      print (t)
      # return Simplex.phase2 (artificial, base)
      return Simplex.phase2 (t)


  # Argument:
  #   t: Tableau of the LPP
  # Return:
  #   Resulting tableau
  @staticmethod
  def phase2(t):
    from .model import Tableau, TYPE, STATUS
    t.pivot_base(t.base)
    print (t)

    while True:
    #for i in range(2):
      # Default: Minimization Problem
      # Find the most negative reduced cost and take the value and it variable
      best, index = min(zip(t.mat[0, :], range (len(t.mat[0]) - 1)))

      # Stop criteria

      # Avoid float point imprecision
      if round(best, 6) >= 0:
        f = t.mat[0, t.mat.n - 1]

        for var, value in zip (t.base, t.mat[1:, t.mat.n - 1]):
          var.set_value (value)

        return t, STATUS.OPTIMAL

      in_var = list(t.labels)[index]

      print (f"{in_var} enters the base; ", end='')

      xB = t.mat[1:, t.mat.n - 1]
      d  = t.mat[1:, index]

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
        elif theta == min_step and t.base[i].type == TYPE.ARTIFICIAL:
          pivot = i

      # Pivoting
      if not pivot == -1:
        out_var = t.base[pivot]
        print (f"{out_var} leaves the base!", end='; ')
        t.base[pivot] = in_var
        print (f"New base: {t.base}")
        t.pivot_base(t.base)
        print ("\n", t)
      else:
        return t, STATUS.UNBOUNDED