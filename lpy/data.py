import random

################################## MATRIX #################################################
class Mat():
  def __init__(self, mat):
    self.mat = mat
    self.m = len (mat)
    self.n = len (mat[0])
    self.dims = (self.n, self.m)

  def __add__ (self, other):
    if isinstance(other, Mat):
      # Check Dimensions
      if not self.dims == other.dims:
        raise Exception(f"Invalid \'+\' operation between \'Mat\' with different dimensions! Found {self.dims} + {other.dims}")

      res = [[0] * self.n for _ in range(self.m)]
      for i in range(self.m):
        for j in range(self.n):
          res[i][j] = self.mat[i][j] + other.mat[i][j]
      return Mat(res)
    else:
      raise Exception(f"Invalid \'+\' operation between \'Mat\' and {type(other).__name__}")

  def __mul__ (self, other):
    if isinstance(other, Mat):
      # Check Dimensions
      if not self.n == other.m:
        raise Exception(f"Invalid \'*\' operation between \'Mat\' with dimensions! Found {self.dims} * {other.dims}")

      res = [[0] * other.n for _ in range(self.m)]
      for i in range(self.m):
        for j in range(other.n):
          for k in range(self.n):
            res[i][j] += self.mat[i][k] * other.mat[k][j]

      return Mat(res)
    else:
      raise Exception(f"Invalid \'*\' operation between \'Mat\' and {type(other).__name__}")

  # For Index accessing
  def __getitem__ (self, index):
    # Checking for slices index (like 1:3, 0)
    if isinstance (index, tuple):
      if len (index) == 2:
        # Checking for slices
        if isinstance(index[0], slice):
            # Get a slice of rows, with specific column
            return [row[index[1]] for row in self.mat[index[0]]]
        
        elif isinstance(index[1], slice):
            # Get a full row, but a slice of columns
            return self.mat[index[0]][index[1]]

        # Receiving a tuple x, y
        return self.mat[index[0]][index[1]]
      else:
        raise Exception(f"Invalid index {index}! Only bidimentional array is supported!")
    elif isinstance (index, int):
      return self.mat[index]
    else:
      raise Exception(f"Invalid index access to the type {type(index).__name__}")
  
  # For index sign
  def __setitem__ (self, index, value):
    if isinstance (index, tuple):
      if len (index) == 2:
        # Receiving a tuple x, y
        self.mat[index[0]][index[1]] = value
      else:
        raise Exception(f"Invalid index {index}! Only bidimensional array is supported!")
    elif isinstance (index, int):
      if isinstance (value, list) and len (value) == self.n:
        self.mat[index] = value
      else:
        raise Exception (f"Row sign must be a 1, {self.n} dimensional array!")
    else:
      raise Exception(f"Invalid index access to the type {type(index).__name__}")

  def __pow__ (self, other):
    if isinstance (other, int):
      # Inverse Matrix
      if other == -1:
        if not self.m == self.n:
          raise Exception(f"The matrix must to be a square matrix! Found dims={self.dims}")
        
        # Calculate the inverse matrix: TODO
    else:
      raise Exception(f"Invalid exponent of type {type(other).__name__}! Just integers allowed!")
  
  def __repr__ (self):
    repr = []
    for row in self.mat:
      repr.append(str(row))
    return "\n".join(repr)
  
  def lu_decompose (self):
    ''' Doolittle method'''
    # Checking square
    if not self.m == self.n:
      raise Exception(f"Invalid Matrix with dimension: {self.dims} for LU decomposition")

    L = self.zeros(self.dims)
    U = self.zeros(self.dims)

    for i in range (self.m):
      # Upper triangular matrix
      for k in range (i, self.n): # Upper iteration (i, k; k = i..n)
        summation = 0
        for j in range (i): # Lower iteration
          summation += (L[i, j] * U[j, k])
        U[i, k] = self[i, k] - summation

      # Lower triangle
      for k in range (i, self.n): # Upper iteration
        if i == k:
          L[i, k] = 1
        else:
          summation = 0
          for j in range (i): # Lower iteration
            summation += (L[k, j] * U[j, i])
          L[k, i] = (self[k, i] - summation)/ U[i, i]
    
    # Returning the results
    return L, U

  # class method = factory methods
  @classmethod
  def I (cls, order):
    return cls([[1 if i == j else 0 for j in range(order)] for i in range(order)])

  @classmethod
  def zeros (cls, dims):
    m, n = dims
    return cls([[0] * n for _ in range(m)])

  @classmethod
  def random (cls, dims, interval=(0, 1), element=float):
    m, n = dims
    generator = None
    if element.__name__ == "int":
      generator = random.randint
    elif element.__name__ == "float":
      generator = random.uniform
    
    if not generator == None:
      # Create a matrix
      a, b = interval
      mat = [[generator(a, b) for _ in range(n)] for _ in range(m)]
      return cls(mat)
    else:
      raise Exception (f"Invalid type {element.__name__}: Use int or float")