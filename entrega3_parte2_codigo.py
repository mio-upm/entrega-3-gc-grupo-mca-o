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
import math

# Datos -----------------------------------------------------------------------
datos = pd.read_excel('241204_datos_operaciones_programadas.xlsx')              # Importación datos desde archivos excel
costes = pd.read_excel('241204_costes.xlsx', index_col=0)                       # Importación costes desde archivos excel

operaciones = ['Cardiología Pediátrica', 'Cirugía Cardíaca Pediátrica', 'Cirugía Cardiovascular', 'Cirugía General y del Aparato Digestivo']
datos_filtrados = datos[datos['Especialidad quirúrgica'].isin(operaciones)]

C_i = {i : costes[i].mean() for i in datos_filtrados['Código operación']}

inicio = datos_filtrados.iloc[0]['Hora inicio']
fin = datos_filtrados.iloc[0]['Hora fin']

inicio > fin

K = [] # Planificaciones factibles

# Modelo del problema ---------------------------------------------------------
# Declaracion del modelo del problema
model = lp.LpProblem(name = "Problema", sense = lp.LpMinimize)                 # Problema de minimización de costes de asignación

# Declarar variables de decisión
y = lp.LpVariable.dicts("y", [k for k in K], cat = lp.LpBinary)                             

# Declaracion de la función objetivo


# Declaracion restricciones



for l in G.edges():
    model += lp.lpSum(x[(l,p)] for p in productos.keys()) <= len(productos.keys())*y[l]

# Tras la resolución e impresión de resultados --------------------------------
model.solve()
print('Estado del problema = ', lp.LpStatus[model.status])

for v in model.variables():
    print(v.name, '=', v.value())
    
print('Objective = ', lp.value(model.objective))