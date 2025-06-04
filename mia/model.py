from .data import Mat
from .algorithms import Simplex
from .algebra import *
from collections import defaultdict

class STATUS(Enum):
  UNSOLVED = 0
  OPTIMAL = 1
  INFEASIBLE = 2
  MULTIPLE = 3
  UNBOUNDED = 4

  def __repr__ (self):
    if self is STATUS.UNSOLVED:
      return "\r\n* [Mia]: Model not yet optimized."
    elif self is STATUS.OPTIMAL:
      return "\r\n* [Mia]: Model has Optimal Solution."
    elif self is STATUS.INFEASIBLE:
      return "\r\n* [Mia]: Infeasible problem."
    elif self is STATUS.UNBOUNDED:
      return "\r\n* [Mia]: Unbouded problem."

class OBJECTIVE:
  MINIMIZE = 0
  MAXIMIZE = 1

# Model
class Model ():
  def __init__ (self):
    self.vars = {}      # Dictionary of variables
    self.constrs = []   # List of Constraints
    self.objective = None
    self.status = STATUS.UNSOLVED

    self.replaces = {}
    
    self.var_count = 0
    self.constr_count = 0

  def add_var (self, var):
    if var.name == None:
      var.name = f"x{Variable._counter}"

    if self.vars.get(var.name) == None:
      self.vars[var.name] = var
      self.var_count += 1

  def add_constr(self, constr):
    self.constrs.append(constr)
    self.constr_count += 1

  def set_objective(self, objective):
    if isinstance (objective, Expression):
      self.objective = objective
    else:
      raise Exception("Invalid objective function!")

  # Convert this model to the standard form and return it
  @classmethod
  def standardize(cls, model):
    # std_model = Model.copy(model)
    std_model = cls()

    for var in model.vars.values():
      var = Variable.copy(var)
      if var.bounds[0] > -float('inf') and not var.bounds[0] == 0:
        std_model.add_constr(var >= var.bounds[0])
        var.bounds = (-float('inf'), var.bounds[1])
      
      if var.bounds[1] < float('inf') and not var.bounds[1] == 0:
        std_model.add_constr(var <= var.bounds[1])
        var.bounds = (var.bounds[0], float('inf'))
      
      # Check the new boundaries
      if var.bounds == (-float('inf'), float('inf')):
        positive_var = Variable(name=f"{var.name}+", type=TYPE.PARTIAL_POSITIVE)
        negative_var = Variable(name=f"{var.name}-", type=TYPE.PARTIAL_NEGATIVE)

        std_model.replaces[var] = (positive_var - negative_var)

        std_model.add_var(positive_var)
        std_model.add_var(negative_var)

      elif var.bounds == (-float('inf'), 0):
        z = Variable(name=f"-({var})", type=TYPE.REPLACE)
        std_model.replaces[var] = -z
        std_model.add_var(z)
      else:
        std_model.add_var(var)

    # Apply the replaces to the variables in all of the context

    if not model.objective == None:
      std_objective = Expression.copy(model.objective)
      replaced = Expression.replace(std_objective, std_model.replaces)
      std_model.set_objective(replaced)

    for constr in model.constrs:
      std_model.add_constr(Constraint.copy(constr))

    # Standardize the constraints
    for constr in std_model.constrs:
      # std_model.add_constr(Constraint(std_terms, constr.sense, constr.resource, name=constr.name))
      constr.expr = Expression.replace(constr.expr, std_model.replaces)
      if constr.resource.coef < 0:
        constr.resource = -constr.resource
        constr.sense = -constr.sense
        constr.expr = -constr.expr

    for index, constr in enumerate(std_model.constrs):
      # Create the slack variable
      slack = Variable(name=f"s{index}", type=TYPE.SLACK)
      if constr.sense == SENSE.LE:
        # Add a positive slack variable
        std_model.add_var(slack)
        constr.expr += slack
        constr.sense = SENSE.EQ

      elif constr.sense == SENSE.GE:
        # Add negative slack variable
        std_model.add_var(slack)
        constr.expr -= slack
        constr.sense = SENSE.EQ

    return std_model 

  # Return all the slack variables
  def get_slack (self):
    ret = list()
    for name, var in self.vars.items():
      if var.type == TYPE.SLACK:
        ret.append(var)
    return ret

  def optimize(self, objective):
    std_model = Model.standardize(self)

    if self.objective == None:
      return Simplex.phase1(Model.copy(std_model))

    if objective == OBJECTIVE.MAXIMIZE:
      std_model.objective = -std_model.objective
    
    # Phase 1: Find a feasible basic solution
    # f, base = Simplex.phase1(std_model)
    print ("Phase 1: ==========================================")
    tableau, status = Simplex.phase1(Model.copy(std_model))
    print ("End Phase 1: ======================================")
    
    print (tableau)

    f = tableau.mat[0, tableau.mat.n - 1]
    
    if not round(f, 6) == 0:
      self.status = STATUS.INFEASIBLE
      return 0, None

    elif not tableau.base:
      self.status = STATUS.UNBOUNDED
      return 0, None

    t = Tableau(std_model, tableau.base)
    # Update the tableu of the artificial problem into the standard problem
    print ("STANDARD MODEL TABLEAU")
    print (t)
    print ("UPDATING TO ARTIFICIAL PROBLEM")
    t.update(tableau)
    print (t)

    # Solution = List of variables
    print ("Phase 2: ==========================================")
    # f, base = Simplex.phase2(std_model, base)
    tableau, status = Simplex.phase2(t)
    print ("End Phase 2: ======================================")

    print (f"Algorithm STATUS: {status}")

    if status == STATUS.UNBOUNDED:
      self.status = STATUS.UNBOUNDED
      return float('inf'), None

    self.status = STATUS.OPTIMAL

    assignment = dict([(var, var.value) for var in tableau.base])
    solution = dict()

    # For each variable in the model
    for var in self.vars.values():
      if std_model.replaces.get(var):
        solution[var] = Expression.apply(std_model.replaces[var], assignment)
      elif assignment.get(var):
        solution[var] = assignment[var]
      else:
        solution[var] = 0
      
    return Expression.apply(self.objective, solution), solution

  def __repr__(self):
    str_obj = f"{self.objective}\n" if self.objective is not None else "\n" 
    str_constrs = "Subject to:\n\t" + "\n\t".join([f"{constr}" for constr in self.constrs])
    str_boundaries = "\n\t".join([f"{var}: {var.bounds}" for var in self.vars.values()])
    return str_obj + str_constrs + "\n\t" + str_boundaries

  @classmethod
  def copy (cls, model):
    cpy_model = cls()
    if not model.objective == None:
      cpy_model.set_objective(Expression.copy(model.objective))

    # Copy the Variables
    for var in model.vars.values():
      cpy_model.add_var(Variable.copy(var))

    # Copy The Constraint
    for constr in model.constrs:
      cpy_model.add_constr(Constraint.copy(constr))

    return cpy_model

###################################### TABLEAU ####################################
class Tableau():
  # Convert the model to the tableau representation
  def __init__(self, model, base=None):
    ''' Enter a model to work with tableau '''

    if not isinstance (model, Model):
      return NotImplemented

    # Converting the Model to Tableau
    # The variables of the tableau
    # self.labels = list(model.vars.values())
    self.labels = dict((key, index) for index, key in enumerate(model.vars.values()))
    self.mat = Mat.zeros((model.constr_count + 1, model.var_count + 1))
    self.base = base if base else list()

    for index, var in enumerate(model.vars.values()):
      # Objective Function
      for terms in model.objective.terms:
        if terms.var.same(var):
          self.mat[0, index] = terms.coef

      # Constraint
      for row, constr in enumerate(model.constrs):
        for term in constr.expr.terms:
          if term.var.same(var):
            self.mat[row + 1, index] = term.coef

    for row, constr in enumerate(model.constrs):
      self.mat[row + 1, model.var_count] = constr.resource.coef

  def pivot_base(self, base):
    pivots = list()
    for row, var in enumerate(base):
      # Respecting the order of the base
      pivots.append ((row + 1, self.labels[var]))

    for pivot in pivots:
      col = pivot[1]
      pivot_col = self.mat[:, col]

      Q = Mat.I(self.mat.m)
      for row, element in enumerate(pivot_col):
        if row == pivot[0] and not pivot_col[row] == 0:
          Q[row, row] /= pivot_col[row]
        else:
          Q[row, pivot[0]] = -float(pivot_col[row]/ self.mat[pivot])
      
      self.mat = Q * self.mat
    self.base = base
    
  # Update the tableau given the variables in the base
  def update(self, other):
    if isinstance(other, Tableau):
      for label, col in self.labels.items():
        if label in other.labels:
          other_col = other.labels[label]
          for i in range(1, self.mat.m):
            self.mat[i, col] = other.mat[i, other_col]

      # Copy the last column (RHS or z-values)
      for i in range(self.mat.m):
        self.mat[i, -1] = other.mat[i, -1]


  def __repr__ (self):
    base = f"{[var for var in self.base]};\n"
    repr = f"{self.labels}\n" + f"{self.mat}\n"
    return base + repr