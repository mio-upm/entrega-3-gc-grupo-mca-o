# -*- coding: utf-8 -*-
"""
Metodos Cuantitativos Avanzados (MIO)

ENTREGA 3: Asignaciòn de quiròfanos en un hospital y generaciòn de columnas
Modelo 3 (Gen. columnas)

Grupo O
- KIANI HANIE
- PAEZ LEAL JAIRO EDUARDO
- PAJARES CAMACHO JAVIER
- TREVISAN FEDERICO
"""

# Importación de librerías ----------------------------------------------------
import pulp as lp                                                              # Para la resolución de problemas de PL 
import pandas as pd
import numpy as np

# Funciones -------------------------------------------------------------------

def operaciones_incompatibles(datos):
    incompatibilidades = {op: set() for op in datos.index}
    for i, op_a in datos.iterrows():
        for j, op_b in datos.iterrows():
            if i != j:
                if (op_a["Hora inicio"] < op_b["Hora fin"]) and (op_b["Hora inicio"] < op_a["Hora fin"]):
                    incompatibilidades[i].add(j)
                    incompatibilidades[j].add(i)
    return incompatibilidades

def master_problem(planificacion):
    
    # Declaracion del modelo del problema
    model_master = lp.LpProblem(name = "Problema", sense = lp.LpMinimize)      # Problema de minimización

    # Declarar variables de decisión
    x = lp.LpVariable.dicts("x", [k for k in planificacion.columns], cat = lp.LpContinuous, lowBound=0)

    # Declaracion de la función objetivo
    model_master += lp.lpSum(x[k] for k in planificacion.columns)

    # Declaracion restricciones
    for i in planificacion.index:
        model_master += lp.lpSum(planificacion.loc[i,k] * x[k] for k in planificacion.columns) >= 1
   
    return model_master

def generacion_columnas(shadow_prices, datos, incompatibilidades):
    
    # Declaracion del modelo del problema
    model_generacion = lp.LpProblem(name = "Problema", sense = lp.LpMaximize)  # Problema de maximización

    # Declarar variables de decisión
    y = lp.LpVariable.dicts("y", [i for i in datos.index], cat = lp.LpBinary)

    # Declaracion de la función objetivo
    model_generacion += lp.lpSum(shadow_prices[i] * y[i] for i in datos.index)

    # Declaracion restricciones
    for i in datos.index:
        model_generacion += y[i] + lp.lpSum(y[j] for j in incompatibilidades[i]) <= 1
        
    return model_generacion


# Datos -----------------------------------------------------------------------
costes = pd.read_excel('241204_costes.xlsx', index_col=0)                       # Importación costes desde archivos excel
datos = pd.read_excel('241204_datos_operaciones_programadas.xlsx', index_col=0) # Importación datos desde archivos excel

# Calcular incompatibilidades basadas en horarios
incompatibilidades = operaciones_incompatibles(datos)

# Planificacion inicial
K_inicial = pd.DataFrame({j: [1 if i != j else 0 for i in range(len(datos.index))] for j in range(len(datos.index))},
                         index=datos.index)




# -----------------------------------------------------------------------------
# master problem
minimizacion_quirofanos = master_problem(K_inicial)

# Tras la resolución e impresión de resultados --------------------------------
minimizacion_quirofanos.solve()
'''
print('Estado del problema = ', lp.LpStatus[master_problem.status])
for v in master_problem.variables():
    print(v.name, '=', v.value()) 
print('Objective = ', lp.value(master_problem.objective))
'''

shadow_prices = {op : c.pi for op in datos.index for name,c in minimizacion_quirofanos.constraints.items()}

# generiacion columnas
nuevo_quirofano = generacion_columnas(shadow_prices, datos, incompatibilidades)

# Tras la resolución e impresión de resultados --------------------------------
nuevo_quirofano.solve()

print('Estado del problema = ', lp.LpStatus[nuevo_quirofano.status])
for v in nuevo_quirofano.variables():
    print(v.name, '=', v.value()) 
print('Objective = ', lp.value(nuevo_quirofano.objective))

