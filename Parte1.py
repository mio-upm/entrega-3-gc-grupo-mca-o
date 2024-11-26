# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:59:15 2024

@author: jpaja
"""
import openpyxl
import pandas as pd
import pulp as lp    

# Lectura operaciones/quirófanos
archivo = "241204_costes.xlsx" 
costesxl = pd.read_excel(archivo)

# Se transpone para que las filas i correspondan a las operaciones (equipos)
# y las filas j correspondan a los quirófanos
costes_transpuesta = costesxl.transpose()

# Se extraen la cantidad de quirófanos existentes
qrfs = costesxl["Unnamed: 0"]

print(costes_transpuesta)

# Lectura datos de operaciones (relaciones operaciones -> equipos -> especialidad -> horas)
archivo2 = "241204_datos_operaciones_programadas.xlsx"
datosxl = pd.read_excel(archivo2)

#dict_por_filas = {index: row.to_dict() for index, row in datosxl.iterrows()}

# Datos de operaciones
ops = datosxl["Código operación"]

# Datos de equipos
eqps = datosxl["Equipo de Cirugía"]

# Equipos filtrados por la especialidad de cardiología pediátrica
equipos_cardio = datosxl.loc[datosxl["Especialidad quirúrgica"] == "Cardiología Pediátrica", "Equipo de Cirugía"]

# Desarrollo del modelo 1
model_1 = lp.LpProblem(name = "Modelo 1", sense = lp.LpMinimize)

# Variable Xij
x = lp.LpVariable.dicts("x", [(i,j) for i in eqps for j in qrfs], lowBound=None, cat=lp.LpBinary)