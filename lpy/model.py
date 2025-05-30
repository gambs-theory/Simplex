from collections import OrderedDict
from .data import Mat
# A library that has all of the classes for optimization (linear)

class Variable():
  # Counting the number of instances
  _counter = 0
  def __init__ (self, bounds=(0, float('inf')), name=None, value=0):
    self.name = name if name is not None else f"x{Variable._counter}"
    self.bounds = bounds
    self.lb = bounds[0]
    self.ub = bounds[1]

    # Store the value
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

  def __repr__(self):
    return self.name

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

  # Constraint: Just in the standard form, for now
  def __le__(self, other):
    if isinstance (other, Expression):
      # # Operation
      # # Left side
      ls = self - other

      # Right side
      index = next((i for i, term in enumerate(ls.terms) if term.var is None), None)
      rs = ls.terms.pop(index) if index else Term(0, None)

      return Constraint(ls, "<=", - rs)

    # Case: x1 + x2 <= -x1
    elif isinstance (other, Variable):
      ls = self - other

      # Right side
      index = next((i for i, term in enumerate(ls.terms) if term.var is None), None)
      rs = ls.terms.pop(index) if index else Term(0, None)

      return Constraint(ls, "<=", -rs)

    # Case: x1 + x2 <= cte
    elif isinstance (other, (int, float)):
      ls = self

      # Right side
      index = next((i for i, term in enumerate(ls.terms) if term.var is None), None)
      rs = ls.terms.pop(index) if index else Term(0, None)

      rs = Term(other, None) - rs 
      return Constraint(ls, "<=", rs)
    
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

      return Constraint(ls, "==", - rs)

    # Case: x1 + x2 <= -x1
    elif isinstance (other, Variable):
      ls = self - other

      # Right side
      index = next((i for i, term in enumerate(ls.terms) if term.var is None), None)
      rs = ls.terms.pop(index) if index else Term(0, None)

      return Constraint(ls, "==", -rs)

    # Case: x1 + x2 <= cte
    elif isinstance (other, (int, float)):
      ls = self

      # Right side
      index = next((i for i, term in enumerate(ls.terms) if term.var is None), None)
      rs = ls.terms.pop(index) if index else Term(0, None)

      rs = Term(other, None) - rs 
      return Constraint(ls, "==", rs)

  def __repr__(self):
    return " + ".join([f"{term}" for term in self.terms])

class Constraint():
  # expr (sense: <=) resource
  def __init__(self, expr, sense, resource, name=None):
    self.name = name
    self.expr = expr    # Expression
    self.sense = sense  
    self.resource = resource # Value (constant) after the singal

  def __repr__(self):
    return f"{self.expr} {self.sense} {self.resource}"

# Model
class Model ():
  def __init__ (self):
    self.vars = {}      # Dictionary of variables
    self.constrs = {}   # Dictionary of Constraints
    self.objective = None

    # Counters
    self.var_count = 0
    self.constr_count = 0

  def add_var (self, var):
    if var.name == None:
      var.name = f"x{self.var_count}"
    
    # Adding variable to the dictionary
    self.vars[var.name] = var
    self.var_count += 1

  def add_constr(self, constr):
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

  # Convert the model to the tableau representation
  def to_mat(self):
    m = Mat.zeros((self.constr_count + 1, self.var_count + 1))

    for index, var in enumerate(self.vars.values()):
      # Objective Function
      for terms in self.objective.terms:
        if terms.var == var:
          m[0, index] = terms.coef

      # print (self.constrs.values())
      # Constraint
      for row, constr in enumerate(self.constrs.values()):
        for term in constr.expr.terms:
          if term.var == var:
            m[row + 1, index] = term.coef

    for row, constr in enumerate(self.constrs.values()):
      m[row + 1, self.var_count] = constr.resource.coef
      
    return m

  def __repr__(self):
    return  (f"{self.objective}\n" if self.objective is not None else "") + "\n".join([f"{name}: {constr}" for name, constr in self.constrs.items()])



###################################### TABLEAU ####################################
class Tableau():
  # Convert the model to the tableau representation
  def __init__(self, model):
    ''' Enter a model to work with tableau '''

    if not isinstance (model, Model):
      return NotImplemented

    # Converting the Model to Tableau
    # The variables of the tableau
    self.labels = list(model.vars.values())
    # Its coef.
    self.mat = Mat.zeros((model.constr_count + 1, model.var_count + 1))

    # Basic variables
    self.base = [None for _ in range(model.constr_count)]
    
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
    repr = f"{self.labels}\n" + f"{self.mat}\n" + f"Base: {self.base}"
    return repr