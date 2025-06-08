from mia.model import *
from mia.data import *
from mia.algorithms import *

import pandas as pd

alimentos = pd.read_csv("./samples/alimentos.csv")

# Recursos
prot = 144    # Proteinas      
carbo = 273   # Carboidratos
gord = 78      # Lipideos

# Funcao objetivo
objetivo = Expression()

# Restricoes
restr_prot = Expression()
restr_carbo = Expression()
restr_gord = Expression()

# Criar o modelo
model = Model()

for index, alimento in alimentos.iterrows():
  # Criar a variavel
  var = Variable(name=alimento['Alimento'])
  
  # Adicionar a variavel no modelo
  model.add_var(var)

  # Adicionar nas restricoes
  restr_prot  += (alimento['Proteinas_g'] * var)
  restr_carbo += (alimento['Carboidratos_g'] * var)
  restr_gord  += (alimento['Gorduras_g'] * var)

  # Nenhum alimento pode passar de 200 g
  model.add_constr(var <= 2)

  # Adicionar na funcao objetivo
  objetivo += var

model.add_constr(restr_prot >= prot)
model.add_constr(restr_carbo >= carbo)
model.add_constr(restr_gord >= gord)
model.add_constr(restr_gord <= 82)

# Restricao de carne
model.add_constr(model.vars['Acem_bovino_moido'] + model.vars['Porco_pernil'] + model.vars['Frango_sobrecoxa_s_pele'] + model.vars['Frango_peito'] <= 1.30)

# Restricoes adicional
model.set_objective(objetivo)

print (model)

f, sol = model.optimize(OBJECTIVE.MINIMIZE)

if model.status == STATUS.OPTIMAL:
  print ("f* =", f)
  for var, value in sol.items():
    print (f"{var} = {value}")

  print (80 * "=")
  for constr in model.constrs:
    print (constr,"Evaluated:",Expression.apply(constr.expr, sol))
else:
  print (model.status)