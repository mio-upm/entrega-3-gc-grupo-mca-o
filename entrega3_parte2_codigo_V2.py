# -*- coding: utf-8 -*-
"""
Metodos Cuantitativos Avanzados (MIO)

ENTREGA 3: Asignaci´on de quir´ofanos en un hospital y generaci´on de columnas
Modelo 2 (homogéneo)

Grupo O
- KIANI HANIE
- PAEZ LEAL JAIRO EDUARDO
- PAJARES CAMACHO JAVIER
- TREVISAN FEDERICO
"""
# Importación de librerías ----------------------------------------------------
import pulp as lp                                                               # Para la resolución de problemas de PL 
import pandas as pd
import numpy as np

# Datos -----------------------------------------------------------------------
costes = pd.read_excel('241204_costes.xlsx', index_col=0)                       # Importación costes desde archivos excel
datos = pd.read_excel('241204_datos_operaciones_programadas.xlsx', index_col=0) # Importación datos desde archivos excel

operaciones = ['Cardiología Pediátrica', 'Cirugía Cardíaca Pediátrica', 'Cirugía Cardiovascular', 'Cirugía General y del Aparato Digestivo']
datos_filtrados = datos[datos['Especialidad quirúrgica'].isin(operaciones)]

# Costo medio di ogni operazione
C_i = {i : costes[i].mean() for i in datos_filtrados.index}



K = {k : [] for k in costes.index}


# Modelo del problema ---------------------------------------------------------
# Declaracion del modelo del problema
model_2 = lp.LpProblem(name = "Problema", sense = lp.LpMinimize)                 # Problema de minimización de costes de asignación

# Declarar variables de decisión
y = lp.LpVariable.dicts("y", [k for k in K], cat = lp.LpBinary)                             

# Declaracion de la función objetivo


# Declaracion restricciones


# Tras la resolución e impresión de resultados --------------------------------
model_2.solve()
print('Estado del problema = ', lp.LpStatus[model_2.status])

for v in model_2.variables():
    print(v.name, '=', v.value())
    
print('Objective = ', lp.value(model_2.objective))