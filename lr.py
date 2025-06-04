# Linear Regression using mia
from mia.model import *
from mia.data import *
from mia.algorithms import *

model = Model()

# Linear Regression
# Dataset
observations = {
  '1': [0.0, 4.0],
  '2': [1.0, 6.5],
  '3': [1.0, 7.5],
  '4': [2.0, 5.5],
  '5': [2.0, 6.0],
  '6': [2.5, 9.5],
  '7': [3.0, 7.0],
  '8': [4.0, 9.0],
}


# Angular and Linear coefficient
a = Variable(name='A', bounds=(-float('inf'), float('inf')))
b = Variable(name='B', bounds=(-float('inf'), float('inf')))

# Adding it to the model
model.add_var(a)
model.add_var(b)

objective = Expression()

for index, observation in observations.items():
  u = Variable(name=f'u{index}', bounds=(0, float('inf')))
  v = Variable(name=f'v{index}', bounds=(0, float('inf')))

  # Adding the variables
  model.add_var(u)
  model.add_var(v)

  # Adding the constraints
  model.add_constr(observation[0] * a + b + (u - v) == observation[1])

  # Objective function building
  objective += (u + v)

model.set_objective(objective)
print (model)

f, sol = model.optimize(OBJECTIVE.MINIMIZE)

if model.status == STATUS.OPTIMAL:
  print ("f* =",f)
  print (f"Y = {sol[a]}x + {sol[b]:.2f}")
else:
  print (model.status)