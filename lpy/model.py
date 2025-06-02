from collections import OrderedDict
from .data import Mat
from .algorithms import Simplex
from collections import defaultdict

# Enumeration of the constraint sense
class SENSE:
  LE = 0
  GE = 1
  EQ = 2

# Enumeration of the variable type
class TYPE:
  DECISION = 0          # Decision variable
  SLACK = 1             # Slack variable
  ARTIFICIAL = 2        # Aritificial variable
  PARTIAL_POSITIVE =  3 # x = (x+ - x-)
  PARTIAL_NEGATIVE = -3 # x = (x+ - x-)

class STATUS:
  UNSOLVED = 0
  OPTIMAL = 1
  INFEASIBLE = 2
  MULTIPLE = 3

class Variable():
  # Counting the number of instances
  _counter = 0
  def __init__ (self, bounds=(0, float('inf')), name=None, type=TYPE.DECISION, value=0):
    self.name = name if name is not None else f"x{Variable._counter}"
    self.bounds = bounds
    self.lb = bounds[0]
    self.ub = bounds[1]
    self.type = type
    self.value = value
    Variable._counter += 1
  
  # Use the name as hash
  def __hash__(self):
    return hash(self.name)

  # Two variables are equals if, and only if, it's has the same NAME
  def __eq__(self, other):
    return isinstance(other, Variable) and self.name == other.name
      
  # Operations: Linear only
  def __add__ (self, other):
    # Summation var + var
    if isinstance (other, Variable):
      # x1 + x1 = (2, x1)
      if self == other:
        return Expression([Term(2, self)])

      # x1 + x2 = {(1, x1) + (1, x2)}
      return Expression([Term(1, self), Term(1, other)])
    
    # Summation var + expr
    elif isinstance (other, Expression):
      # Let Expresison handle it
      return other + self

    # Summation var + cte
    elif isinstance (other, (int, float)):
      return Expression([Term(1, self), Term(other, None)])

  def __radd__(self, other):
    return self.__add__(other)

  def __rmul__ (self, other):
    # Multiplication cte * x1
    if isinstance (other, (int, float)):
      cte = other
      return Expression([Term(cte, self)])
    else:
      return NotImplemented

  # Negative representation definition
  def __neg__ (self):
    return Expression([Term(-1, self)])
    
  def __sub__(self, other):
    # Var - Var
    return self + (-other)

  def __rsub__(self, other):
    return other + (-self)

  def __le__ (self, other):
    if isinstance (other, (int, float)):
      return Constraint(Expression([Term(1, self)]), SENSE.LE, Term(other, None))
    elif isinstance (other, Variable):
      return Constraint(self + other, SENSE.LE, Term(0, None))
    else:
      return NotImplemented
    

  def __repr__(self):
    return self.name

  def set_value(self, value):
    self.value = value

# A Pair Coeficient and Variable  
class Term():
  def __init__ (self, coef, var):
    self.coef = coef
    self.var = var

   # Use the name as hash
  def __hash__(self):
    return hash(self.var)

  # Two variables are equals if, and only if, it's has the same NAME
  def __eq__(self, other):
    return isinstance(other, Variable) and self.var == other.var

  def __add__ (self, other):
    if isinstance (other, (int, float)):
      return Expression([self, Term(other, None)])
    # Variable and Term
    elif isinstance (other, Variable):
      if self.var == other:
        return Term(self.coef + 1, other)
      else:
        return Expression([self, Term(1, other)])

    elif isinstance (other, Term):
      # Term involving the same variable
      if self.var == other.var:
        return Term(self.coef + other.coef, self.var)
      else:
        return Expression([self, other])

    elif isinstance (other, Expression):
      for i in range(other):
        if self == other[i]:
          other[i] += self
    
  def __neg__(self):
    return Term(-1 * self.coef, self.var)

  def __sub__(self, other):
    # Var - Var
    return self + (-other)

  def __rsub__(self, other):
    return other + (-self)

  # Just scalar multiplication
  def __rmul__(self, other):
    if isinstance(other, (int, float)):
      return Term(other * self.coef, self.var)
    else:
      return NotImplemented
      
  def __repr__(self):
    # Is a Constant
    if self.var == None:
      return f"{self.coef}"

    return f"{self.coef}*{self.var}"

# Collection of variables and the resources (Ax = b)
class Expression(): 
  # expr: x1 + 2 * x2: [(1, x1), (2, x2)]
  def __init__ (self, terms):
    # terms is a list of tuples mapping the coef and the variable
    # self.terms = terms if terms is not None else []

    # Crating the term as a set
    self.terms = terms if terms is not None else list()

  def __add__ (self, other):
    # Just "append" the variable in ther expression
    if isinstance(other, Variable):
      for i in range(len(self.terms)):
        if other == self.terms[i].var:
          self.terms[i] += Term(1, other)
          return Expression(self.terms)
      return Expression(self.terms + [Term(1, other)])

    if isinstance(other, Term):
      # Check if a term with the same variable already exists in the expression
      for i in range (len(self.terms)):
        if other.var == self.terms[i].var:
          self.terms[i] += other
          return Expression(self.terms)
      return Expression(self.terms + [other])
      
    # Adding both
    elif isinstance(other, Expression):
      var_table = OrderedDict()
      # print (self.terms + other.terms)
      
      for term in self.terms + other.terms:
        if term.var in var_table:
          var_table[term.var] = var_table[term.var] + term
        else:
          var_table[term.var] = term
        # print (var_table)

      return Expression(list(var_table.values()))

    # Add the constant term
    elif isinstance(other, (int, float)):
      # return Expression(self.terms, self.cte + other)
      return Expression(self.terms + [Term(other, None)])

  def __radd__(self, other):
    return self + other

  def __neg__(self):
    # Inverse all of the coeficients
    return Expression([-term for term in self.terms])

  def __sub__(self, other):
    return self + (-other)

  def __rsub__(self, other):
    return (-self) + other

  def __radd__(self, other):
    return self.__add__(other)

  # Just linear operation: Scalar multiplication
  def __rmul__ (self, other):
    # Multiplication cte * x1
    if isinstance (other, (int, float)):
      return Expression([other * term for term in self.terms])
    else:
      return NotImplemented

  # Return the term associated to a varible or a index
  def __getitem__ (self, key):
    if isinstance (key, Variable):
      for term in self.terms:
        if term.var == key:
          return term
      return None
    elif isinstance (key, int):
      if key < 0 or key >= len (self.terms):
        return None
      return self.terms[key]
    elif isinstance (key, str):
      for term in self.terms:
        if term.var.name == key:
          return term
      return None
    else:
      return NotImplemented

  # Constraint: Just in the standard form, for now
  def __le__(self, other):
    if isinstance (other, Expression):
      # # Operation
      # # Left side
      ls = self - other

      # Right side
      index = next((i for i, term in enumerate(ls.terms) if term.var is None), None)
      rs = ls.terms.pop(index) if index else Term(0, None)

      return Constraint(ls, SENSE.LE, - rs)

    # Case: x1 + x2 <= -x1
    elif isinstance (other, Variable):
      ls = self - other

      # Right side
      index = next((i for i, term in enumerate(ls.terms) if term.var is None), None)
      rs = ls.terms.pop(index) if index else Term(0, None)

      return Constraint(ls, SENSE.LE, -rs)

    # Case: x1 + x2 <= cte
    elif isinstance (other, (int, float)):
      ls = self

      # Right side
      index = next((i for i, term in enumerate(ls.terms) if term.var is None), None)
      rs = ls.terms.pop(index) if index else Term(0, None)

      rs = Term(other, None) - rs 
      return Constraint(ls, SENSE.LE, rs)
  
  def __ge__(self, other):
    if isinstance (other, Expression):
      # # Operation
      # # Left side
      ls = self - other

      # Right side
      index = next((i for i, term in enumerate(ls.terms) if term.var is None), None)
      rs = ls.terms.pop(index) if index else Term(0, None)

      return Constraint(ls, SENSE.GE, - rs)

    # Case: x1 + x2 <= -x1
    elif isinstance (other, Variable):
      ls = self - other

      # Right side
      index = next((i for i, term in enumerate(ls.terms) if term.var is None), None)
      rs = ls.terms.pop(index) if index else Term(0, None)

      return Constraint(ls, SENSE.GE, -rs)

    # Case: x1 + x2 <= cte
    elif isinstance (other, (int, float)):
      ls = self

      # Right side
      index = next((i for i, term in enumerate(ls.terms) if term.var is None), None)
      rs = ls.terms.pop(index) if index else Term(0, None)

      rs = Term(other, None) - rs 
      return Constraint(ls, SENSE.GE, rs)
    
  # def __ge__(self, other):
  #   return Constraint(self - other, ">=")

  def __eq__(self, other):
    if isinstance (other, Expression):
      # # Operation
      # # Left side
      ls = self - other

      # Right side
      index = next((i for i, term in enumerate(ls.terms) if term.var is None), None)
      rs = ls.terms.pop(index) if index else Term(0, None)

      return Constraint(ls, SENSE.EQ, - rs)

    # Case: x1 + x2 <= -x1
    elif isinstance (other, Variable):
      ls = self - other

      # Right side
      index = next((i for i, term in enumerate(ls.terms) if term.var is None), None)
      rs = ls.terms.pop(index) if index else Term(0, None)

      return Constraint(ls, SENSE.EQ, -rs)

    # Case: x1 + x2 <= cte
    elif isinstance (other, (int, float)):
      ls = self

      # Right side
      index = next((i for i, term in enumerate(ls.terms) if term.var is None), None)
      rs = ls.terms.pop(index) if index else Term(0, None)

      rs = Term(other, None) - rs 
      return Constraint(ls, SENSE.EQ, rs)

  def __repr__(self):
    return " + ".join([f"{term}" for term in self.terms])

  # Hypothesis: An expression has ONLY 1 slack variable!!!!!
  def get_slack (self):
    for term in self.terms:
      if term.var.type == TYPE.SLACK:
        return term
    return None

class Constraint():
  # expr (sense: <=) resource
  def __init__(self, expr, sense, resource, name=None):
    self.name = name
    self.expr = expr    # Expression
    self.sense = sense  
    self.resource = resource # Value (constant) after the singal

  def __repr__(self):
    if self.sense == SENSE.EQ:
      return f"{self.expr} = {self.resource}"
    elif self.sense == SENSE.LE:
      return f"{self.expr} <= {self.resource}"
    elif self.sense == SENSE.GE:
      return f"{self.expr} >= {self.resource}"

  # Negation of a constraint
  def __neg__(self):
    if self.sense == SENSE.LE:
      return Constraint(-1 * self.expr, SENSE.GE, -1 * self.resource)
    elif self.sense == SENSE.GE:
      return Constraint(-1 * self.expr, SENSE.LE, -1 * self.resource)
    else:
      return Constraint(-1 * self.expre, self.sense, -1 * self.resource)

  # Create a copy of the constraint
  @classmethod
  def copy(cls, constr):
    return cls(constr.expr, constr.sense, constr.resource, constr.name)

# Model
class Model ():
  def __init__ (self):
    self.vars = {}      # Dictionary of variables
    self.constrs = {}   # Dictionary of Constraints
    self.objective = None
    self.status = STATUS.UNSOLVED
    # Counters
    self.var_count = 0
    self.constr_count = 0

  # Adding variable to the model
  def add_var (self, var):
    if var.name == None:
      # Can be hacked!
      var.name = f"x{self.var_count}"

    # Verify variables that already exists: TODO
    if self.vars.get(var.name) == None:
      # Checking boundaries
      # It has negative lower bound? x >= -cte
      if var.bounds[0] < 0:
        self.add_constr(Expression([Term(1, var)]) >= var.bounds[0])
        var.bounds = (-float('inf'), var.bounds[1])

      # Negative upper bound x <= -cte
      if var.bounds[1] < float('inf'):
        # Add a constraing
        # Make it a free variable
        self.add_constr(Expression([Term(1, var)]) <= var.bounds[1])
        var.bounds = (var.bounds[0], float('inf'))

      # Adding variable to the dictionary
      self.vars[var.name] = var
      self.var_count += 1

  def add_constr(self, constr):
    # Checking constraints with resource vector negative
    if constr.resource.coef < 0:
      constr = -constr

    if constr.name == None:
      constr.name = f"C{self.constr_count}"

    # Adding the constraint to the dictionary
    self.constrs[constr.name] = constr
    self.constr_count += 1

  def set_objective(self, objective):
    if isinstance (objective, Expression):
      self.objective = objective
    else:
      raise Exception("Invalid objective function!")

  # Convert this model to the standard form and return it
  def to_standard(self):
    std_model = Model()
    
    if not self.objective == None:
      std_objective = Expression(None)
      # Objective function
      for term in self.objective.terms:
        if term.var.bounds == (-float('inf'), float('inf')):
          # Split it into two variables
          positive_var = Variable(name=f"{term.var.name}+", type=TYPE.PARTIAL_POSITIVE)
          negative_var = Variable(name=f"{term.var.name}-", type=TYPE.PARTIAL_NEGATIVE)

          std_objective += term.coef * (positive_var - negative_var)
        else:
          std_objective += term
      
      std_model.set_objective(std_objective)

    for index, constr in enumerate(self.constrs.values()):
      # For each term of the expression of the constraint
      std_terms = Expression (None)

      for term in constr.expr.terms:
        if term.var.bounds == (-float('inf'), float('inf')):
          # Split it into two variables
          positive_var = Variable(name=f"{term.var.name}+", type=TYPE.PARTIAL_POSITIVE)
          negative_var = Variable(name=f"{term.var.name}-", type=TYPE.PARTIAL_NEGATIVE)

          std_terms += term.coef * (positive_var - negative_var)

          # Add both variables in the model
          std_model.add_var(positive_var)
          std_model.add_var(negative_var)
          
        else:
          std_terms += term
          std_model.add_var(term.var)

      std_model.add_constr(Constraint(std_terms, constr.sense, constr.resource, name=constr.name))

    # For each constraing
    for index, constr in enumerate(std_model.constrs.values()):
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

  # Optimize function
  def optimize(self, minimize=True):
    if self.objective == None:
      return Simplex.phase1(self.to_standard())

    if not minimize:
      self.objective = -self.objective

    std_model = self.to_standard()
    
    # Phase 1: Find a feasible basic solution
    print (Simplex.phase1(std_model))
    fitness, bfs = Simplex.phase1(std_model)

    if not round(fitness, 6) == 0:
      print ("Infeasible")
      return 0, None
    elif bfs == None:
      print ("Ubounded")
      return 0, None

    fitness, solution = Simplex.phase2(std_model, bfs)
    summary = defaultdict(float)

    if solution == None:
      return float('inf'), None

    # For each variable in the base
    for index, var in enumerate(solution):
      if var.type == TYPE.PARTIAL_POSITIVE:
        var_label = var.name[:len(var.name) - 1]
        summary[var_label] += var.value
      elif var.type == TYPE.PARTIAL_NEGATIVE:
        var_label = var.name[:len(var.name) - 1]
        summary[var_label] -= var.value
      elif var.name in self.vars.keys():
        summary[var.name] = var.value
    
    return abs(fitness) if not minimize else -abs(fitness), dict (summary)

  def __repr__(self):
    return  (f"{self.objective}\n" if self.objective is not None else "") + "\t" + "\n\t".join([f"{name}: {constr}" for name, constr in self.constrs.items()])

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

    for index, var in enumerate(model.vars.values()):
      # Objective Function
      for terms in model.objective.terms:
        if terms.var == var:
          self.mat[0, index] = terms.coef

      # print (self.constrs.values())
      # Constraint
      for row, constr in enumerate(model.constrs.values()):
        for term in constr.expr.terms:
          if term.var == var:
            self.mat[row + 1, index] = term.coef

    for row, constr in enumerate(model.constrs.values()):
      self.mat[row + 1, model.var_count] = constr.resource.coef

  def __repr__ (self):
    repr = f"{self.labels}\n" + f"{self.mat}\n"
    return repr

  # Update the tableau given the variables in the base
  def update(self, base):
    pivots = list()
    for row, var in enumerate(base):
      pivots.append ((row + 1, self.labels[var]))

    # print (pivots)
    for pivot in pivots:
      col = pivot[1]
      pivot_col = self.mat[:, col]
      
      # Create the identity matrix
      Q = Mat.I(self.mat.m)
      for row, element in enumerate(pivot_col):
        if row == pivot[0] and not pivot_col[row] == 0:
          Q[row, row] /= pivot_col[row]
        else:
          Q[row, pivot[0]] = -float(pivot_col[row]/ self.mat[pivot])
      
      # print (Q, end='\n' + (80 * '-') + '\n')
      self.mat = Q * self.mat