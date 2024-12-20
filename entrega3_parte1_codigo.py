# -*- coding: utf-8 -*-
"""----------------------------------------------------------------------------
Metodos Cuantitativos Avanzados (MIO)

ENTREGA 3: Asignaci´on de quir´ofanos en un hospital y generaci´on de columnas
Modelo 1

Grupo O
- KIANI HANIE
- PAEZ LEAL JAIRO EDUARDO
- PAJARES CAMACHO JAVIER
- TREVISAN FEDERICO
----------------------------------------------------------------------------"""
# Importación de librerías ----------------------------------------------------
import pulp as lp                                                              # Para la resolución de problemas de PL 
import pandas as pd                                                            # Para trabajar con DataFrames


# Datos -----------------------------------------------------------------------

# Lectura operaciones/quirófanos
costesxl = pd.read_excel("241204_costes.xlsx", index_col=0)

# Se transpone para que las filas i correspondan a las operaciones (equipos)
# y las filas j correspondan a los quirófanos
costes_transpuesta = costesxl.T

# Se extraen la cantidad de quirófanos existentes
qrfs = costes_transpuesta.columns

print(costes_transpuesta)

# Lectura datos de operaciones (relaciones operaciones -> equipos -> especialidad -> horas)
datosxl = pd.read_excel("241204_datos_operaciones_programadas.xlsx")

#dict_por_filas = {index: row.to_dict() for index, row in datosxl.iterrows()}

# Verificar los tipos de datos de cada columna: datosxl.dtypes

# Datos de operaciones
ops = datosxl["Código operación"]

# Datos de equipos
eqps = datosxl["Equipo de Cirugía"]

# Equipos filtrados por la especialidad de cardiología pediátrica
ops_cardio = datosxl.loc[datosxl["Especialidad quirúrgica"] == "Cardiología Pediátrica", "Código operación"]
eqps_cardio = datosxl.loc[datosxl["Especialidad quirúrgica"] == "Cardiología Pediátrica", "Código operación"]

# datosxl.loc[0]["Hora inicio "] Esta columna está escrita con un espacio al final

incompatibles = []

for i in range(0,len(eqps)-1):
    for k in range(0,len(eqps)-1):
        if (datosxl.loc[k]["Hora fin"] >= datosxl.loc[i]["Hora inicio"]) and (datosxl.loc[i]["Hora fin"] >= datosxl.loc[k]["Hora inicio"]) and (i!=k):
            incompatibles.append({'operación': datosxl.loc[i]["Código operación"], 
                                  'incomp': datosxl.loc[k]["Código operación"], 
                                  'equipo op': datosxl.loc[i]["Equipo de Cirugía"],
                                  'equipo incomp': datosxl.loc[k]["Equipo de Cirugía"]})
df = pd.DataFrame(incompatibles)

# df[df['operación'] == '20241204 OP-133']['incomp']

# Crear una nueva columna con los pares ordenados
df['Par_Ordenado'] = df[['operación', 'incomp']].min(axis=1) + '_' + df[['operación', 'incomp']].max(axis=1)

# Eliminar duplicados en la columna 'Par_Ordenado'
# Revisar si está bien este código, de momento funciona
df = df.drop_duplicates(subset='Par_Ordenado')
df.drop('Par_Ordenado', axis=1, inplace=True)

# De la tabla df filtrar si ambos equipos pertenecen a cardiología pediátrica
# Data final, con la que se trabaja el modelo 1
# df_filtrado = df[(df['equipo op'].isin(eqps_cardio)) & (df['equipo incomp'].isin(equipos_cardio))]
df_filtrado = df[(df['operación'].isin(ops_cardio)) & (df['incomp'].isin(ops_cardio))]


# Modelo del problema ---------------------------------------------------------
# Declaracion del modelo del problema
model_1 = lp.LpProblem(name = "Modelo 1", sense = lp.LpMinimize)

# Declaracion variables de decisión
x = lp.LpVariable.dicts("x", [(i,j) for i in ops_cardio for j in qrfs], lowBound=None, cat=lp.LpBinary)

# Declaracion de la función objetivo
model_1 += lp.lpSum(costes_transpuesta.loc[i][j]*x[(i,j)] for i in ops_cardio for j in qrfs)   # Costes variables

# Declaracion restricciones
for i in ops_cardio: 
    model_1 += lp.lpSum(x[(i,j)] for j in qrfs) >= 1                           # Restricción cada op tiene que estar asignada a un quirófano

for i in ops_cardio:                                                           # Restricción dos op no pueden ser incompatibles
    for j in qrfs:                                                             # Es decir que pueden ser asignadas al mismo quirófano pero no en la misma hora
        for h in df_filtrado[df_filtrado['operación'] == i]['incomp']:
            model_1 += lp.lpSum(x[(h,j)] + x[(i,j)]) <= 1
  
# Tras la resolución e impresión de resultados
model_1.solve()
print('Estado del problema = ', lp.LpStatus[model_1.status])

# Guardar variables
valores_variables = {v.name: v.varValue for v in model_1.variables() if v.varValue == 1}

for v in model_1.variables():
    print(v.name, '=', v.value())
    
print('Objective = ', lp.value(model_1.objective))
