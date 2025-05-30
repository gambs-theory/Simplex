#from lpy import Variable, Term, Constraint, Model
#from methods import Simplex
#from mat import Mat

from lpy.data import Mat
from lpy.model import Variable, Model, Tableau
from lpy.algorithms import Simplex

x1 = Variable(name='x')
x2 = Variable(name='y')

# Slack variable
s1 = Variable(name='s1')
s2 = Variable(name='s2')
s3 = Variable(name='s3')

model = Model()

model.add_var(x1)
model.add_var(x2)

model.add_var(s1)
model.add_var(s2)
model.add_var(s3)

model.add_constr( x1 + x2 - s1 == 1)
model.add_constr(-x1 + x2 + s2 == 3)
model.add_constr( x1 + x2 + s3 == 7)

model.set_objective(2 * x1 + x2)
# print (Simplex.solve(Tableau.model_to_tableau(model)))
t = Tableau(model)

Simplex.solve(t)