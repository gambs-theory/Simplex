from lpy.model import *

model = Model()

aveia = Variable(name="aveia")
leite = Variable(name="leite")
banana = Variable(name="banana")
arroz = Variable(name="arroz")
feijao = Variable(name="feijao")
frango = Variable(name="frango")
queijo = Variable(name="queijo")
ovo = Variable(name="ovo")
pao_frances = Variable(name="pao_frances")
pasta_amendoim = Variable(name="pasta_amendoim")
whey = Variable(name="whey")

model.add_var(aveia)
model.add_var(leite)
model.add_var(banana)
model.add_var(arroz)
model.add_var(feijao)
model.add_var(frango)
model.add_var(queijo)
model.add_var(ovo)
model.add_var(pao_frances)
model.add_var(pasta_amendoim)
model.add_var(whey)

# 1.44 g de proteinas por dia
#  273 g de proteinas por dia
#   78 g de proteinas por dia
proteinas = 144      
carboidratos = 273    
lipidios = 78         

# 104 gramas de leite eh 100 ml

# Proteinas
model.add_constr(10 * aveia + 3 * leite + 1.3 * banana + 65 * whey + 2.4 * arroz + 13.6 * feijao + 32 * frango + 17.3 * queijo + 8.4 * pao_frances + 27.3 * pasta_amendoim + 12.6 * ovo >= proteinas)
# model.add_constr(13.6 * arroz + 12.6 * ovo + 13.6 * feijao >= proteinas)

# Carboidratos
model.add_constr(60 * aveia + 4.7 * leite + 26 * banana + 28 * arroz + 13.6 * feijao + 3.2 * queijo + 1.2 * ovo + 60 * pao_frances + 20 * pasta_amendoim + 19.8 * whey >= carboidratos)
# model.add_constr(28 * arroz + 1.2 * ovo  + 13.6 * feijao >= carboidratos)

# Lipidios
model.add_constr(10 * aveia + 3.1 * leite + 0.1 * banana + 4.7 * whey + 0.2 * arroz + 0.5 * feijao + 2.5 * frango + 20.2 * queijo + 2 * pao_frances + 46.7 * pasta_amendoim + 10 * ovo >= lipidios)
# model.add_constr(0.2 * arroz + 10 * ovo + 0.5 * feijao >= lipidios)

model.add_constr(arroz >= 1)
model.add_constr(whey <= 0.4)
model.add_constr(pasta_amendoim <= 0.3)
model.add_constr(aveia <= 1)
model.add_constr(leite >= 3)

# print (Simplex.solve(Tableau.model_to_tableau(model)))
model.set_objective(aveia + leite + banana + whey + arroz + feijao + frango + ovo + pasta_amendoim + pao_frances + queijo)
# model.set_objective(arroz + ovo)

fitness, sol = model.optimize(OBJECTIVE.MINIMIZE)

print (repr(model.status))

print ("f* =", fitness, ";", sol)