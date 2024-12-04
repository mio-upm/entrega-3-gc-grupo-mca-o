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

# Funciones -------------------------------------------------------------------

#hay que ordenar operaciones 
def nueva_planificacion(operaciones, costes):  #dataframes con operac ordenadas
    #objectivo es generar un conjunto de planificacion K
    quirofanos = costes.index
    K = { q: [] for q in quirofanos } #vamos a poner las operaciones de cada uno
    activados = []  #quirofanos activados
    num_activados = 0
    for codigo,op in operaciones.iterrows():
        t_inicio = op["Hora inicio"]
        t_fin = op["Hora fin"]
        # asigna operacion al primero quirofano disponible
        asignado = False

        i=0
        while i < len(activados) and not asignado:
            quiro = activados[i]
            if K[quiro][-1][2] <= t_inicio:
               #libre
               asignado = True
               K[quiro].append( (codigo, t_inicio, t_fin) )
            else: i+=1
        
        if not asignado: #nuevo
            asignado = True
            quiro = quirofanos[num_activados]
            K[quiro] = [ (codigo, t_inicio, t_fin) ]
            activados.append(quiro)
            num_activados+=1
    return K            # contenedor K = { quirofanos : [(op1, inicio, fin), (op2...)]   }

# Datos -----------------------------------------------------------------------
costes = pd.read_excel('241204_costes.xlsx', index_col=0)                       # Importación costes desde archivos excel
datos = pd.read_excel('241204_datos_operaciones_programadas.xlsx', index_col=0) # Importación datos desde archivos excel

operaciones = ['Cardiología Pediátrica', 'Cirugía Cardíaca Pediátrica', 'Cirugía Cardiovascular', 'Cirugía General y del Aparato Digestivo']
datos_filtrados = datos[datos['Especialidad quirúrgica'].isin(operaciones)]

# K  
planificaciones = nueva_planificacion(datos_filtrados, costes)

     
'''
# Mostrar resultados
for idx, quir in enumerate(resultados):
    print(f"Quirófano {idx + 1}:")
    for op in quir:
        print(f"  {op['id']} - Inicio: {op['inicio']} - Fin: {op['fin']}")
'''

# Costo medio di ogni operazione
C_i = {i : costes[i].mean() for i in datos_filtrados.index}

B_ik = {}

for i in datos_filtrados.index:
    for k in planificaciones.keys():
        long=len(planificaciones[k])
        lista_operaciones = [planificaciones[k][j][0] for j in range (long)]
        if i in lista_operaciones:
            B_ik[(i,k)] = 1 
        else: 
            B_ik[(i,k)] = 0

  
C_k={}

for k in planificaciones.keys():
    C_k[k]=0
    for i in datos_filtrados.index:
        C_k[k]=C_k[k]+round(C_i[i]*B_ik[(i,k)],2)

# Modelo del problema ---------------------------------------------------------
# Declaracion del modelo del problema
model_2 = lp.LpProblem(name = "Problema", sense = lp.LpMinimize)                 # Problema de minimización de costes de asignación

# Declarar variables de decisión
y = lp.LpVariable.dicts("y", [k for k in planificaciones.keys()], cat = lp.LpBinary)

# Declaracion de la función objetivo
model_2 += lp.lpSum(C_k[(k)]*y[(k)] for k in planificaciones.keys())

# Declaracion restricciones
for i in datos_filtrados.index:
    model_2 += lp.lpSum(B_ik[(i, k)] * y[k] for k in planificaciones.keys()) >= 1


# Tras la resolución e impresión de resultados --------------------------------
model_2.solve()
print('Estado del problema = ', lp.LpStatus[model_2.status])

for v in model_2.variables():
    print(v.name, '=', v.value())
    
print('Objective = ', lp.value(model_2.objective))