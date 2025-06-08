from enum import Enum
from collections import OrderedDict

# Enumeration of the constraint sense
class SENSE(Enum):
  LE = 0
  GE = 1
  EQ = 2

  def __repr__ (self):
    if self is SENSE.LE:
      return "<="
    elif self is SENSE.EQ:
      return "=="
    elif self is SENSE.GE:
      return ">="
  
  def __neg__ (self):
    if self is SENSE.LE:
      return SENSE.GE
    elif self is SENSE.GE:
      return SENSE.LE
    else:
      return self

# Enumeration of the variable type
class TYPE:
  DECISION = 0          # Decision variable
  SLACK = 1             # Slack variable
  ARTIFICIAL = 2        # Aritificial variable
  PARTIAL_POSITIVE =  3 # x = (x+ - x-)
  PARTIAL_NEGATIVE = -3 # x = (x+ - x-)
  REPLACE = 4

class Variable():
  # Counting the number of instances
  _counter = 0

  def __init__ (self, bounds=(0, float('inf')), name=None, type=TYPE.DECISION, value=0):
    self.name = name if name is not None else f"x{Variable._counter + 1}"
    self.bounds = bounds
    self.type = type
    self.value = value
    Variable._counter += 1
  
  # Use the name as hash
  def __hash__(self):
    return hash(self.name)
      
  # Operations: Linear only
  def __add__ (self, other):
    # Summation var + var
    if isinstance (other, Variable):
      # If the variables are the same
      if self.same(other):
        return Expression([Term(2, self)])

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

  def __neg__ (self):
    return Expression([Term(-1, self)])
    
  def __sub__(self, other):
    return self + (-other)

  def __rsub__(self, other):
    return other + (-self)

  def __le__ (self, other):
    if isinstance (other, (int, float)):
      return Constraint(Expression([Term(1, self)]), SENSE.LE, Term(other, None))
    elif isinstance (other, Variable):
      return Constraint(self - other, SENSE.LE, Term(0, None))
    else:
      return NotImplemented

  def __ge__ (self, other):
    if isinstance (other, (int, float)):
      return Constraint(Expression([Term(1, self)]), SENSE.GE, Term(other, None))
    elif isinstance (other, Variable):
      return Constraint(self - other, SENSE.GE, Term(0, None))
    else:
      return NotImplemented

  def __eq__ (self, other):
    if isinstance (other, (int, float)):
      return Constraint(Expression([Term(1, self)]), SENSE.EQ, Term(other, None))
    elif isinstance (other, Variable):
      return Constraint(self - other, SENSE.EQ, Term(0, None))
    else:
      return NotImplemented

  def __repr__(self):
    return self.name if self.type == TYPE.DECISION else f"\"{self.name}\""

  @classmethod
  def copy (cls, var):
    return cls(bounds=var.bounds, type=var.type, name=var.name, value=var.value)

  def same(self, other):
    if isinstance (other, Variable):
      return self.name == other.name and self.bounds == other.bounds and self.type == other.type
    return False

  def set_value(self, value):
    self.value = value

class Term():
  ''' * Terms with var = None are Constant '''

  def __init__ (self, coef, var):
    self.coef = coef
    self.var = var

  def __hash__(self):
    return hash(self.var)

  def __eq__(self, other):
    return isinstance(other, Variable) and self.var.same(other.var)

  def __add__ (self, other):
    if isinstance (other, (int, float)):
      return Expression([self, Term(other, None)])
    
    elif isinstance (other, Variable):
      if self.var.same(other):
        # Just incremet it coefficient
        return Term(self.coef + 1, other)
      else:
        return Expression([self, Term(1, other)])

    elif isinstance (other, Term):
      if self.var == None and other.var == None:
        return Term(self.coef + other.coef, None)

      elif (self.var == None and not other.var == None) or (not self.var == None and other.var == None):
        return Expression([other, self])

      if self.var.same(other.var):
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

class Expression(): 
  def __init__ (self, terms=None):
    self.terms = terms if terms is not None else list()

  def __add__ (self, other):
    # Just "append" the variable in ther expression
    if isinstance(other, Variable):
      for i in range(len(self.terms)):
        if other.same(self.terms[i].var):
          self.terms[i] += Term(1, other)
          return Expression(self.terms)
      return Expression(self.terms + [Term(1, other)])

    if isinstance(other, Term):
      # Check if a term with the same variable already exists in the expression
      for i in range (len(self.terms)):
        if other.var.same(self.terms[i].var):
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
        
      return Expression(list(var_table.values()))

    elif isinstance(other, (int, float)):
      return Expression(self.terms + [Term(other, None)])

  def __radd__(self, other):
    return self + other

  def __neg__(self):
    return Expression([-term for term in self.terms])

  def __sub__(self, other):
    return self + (-other)

  def __rsub__(self, other):
    return (-self) + other

  def __radd__(self, other):
    return self.__add__(other)

  # Just linear operation: Scalar multiplication
  def __rmul__ (self, other):
    if isinstance (other, (int, float)):
      return Expression([other * term for term in self.terms])
    else:
      return NotImplemented

  def __getitem__ (self, key):
    if isinstance (key, Variable):
      for term in self.terms:
        if term.var.same(key):
          return term
      return None
    elif isinstance (key, int):
      if key < 0 or key >= len (self.terms):
        return None
      return self.terms[key]
    else:
      return None

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

  # Replace a variable for an expression
  @classmethod
  def replace(cls, expr, replace: dict):
    replaced = cls()
    for term in expr.terms:
      if replace.get(term.var):
        replaced += (term.coef * replace[term.var])
      else:
        replaced += term
    return replaced

  # values = map between the variable and it value
  @staticmethod
  def apply(expr, values: dict) -> float:
    result = float (0)
    for term in expr.terms:
      if values.get(term.var):
        result += term.coef * values.get(term.var)
    return result

  @classmethod
  def copy(cls, expr):
    return cls(terms=expr.terms)

##################################### CONSTRAINT ##############################################
class Constraint():
  def __init__(self, expr, sense, resource, name=None):
    self.name = name
    self.expr = expr    # Expression
    self.sense = sense  
    self.resource = resource # Value (constant) after the singal

  def __repr__(self):
    return f"{self.expr} {repr(self.sense)} {self.resource}"

  # Negation of a constraint
  def __neg__(self):
    return Constraint (-1 * self.expr, -self.sense, -1 * self.resource)

  # Create a copy of the constraint
  @classmethod
  def copy(cls, constr):
    return cls(constr.expr, constr.sense, constr.resource, constr.name)