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
import pulp as lp                                                              # Para la resolución de problemas de PL 
import pandas as pd                                                            # Para trabajar con DataFrames


# Funciones -------------------------------------------------------------------

# Funcion para la generacion de una planificacion factible
def nueva_planificacion(datos, costes):
    quirofanos = [q for q in costes.index]                                     # Listas de los quirófanos
    K_dict = {q: [] for q in quirofanos}                                       # 'claves' = quirófanos ; 'valores' = operaciones asignadas a los quirófanos
    q_activos = []                                                             # Quirófanos activados
    asignaciones = []                                                          # Para construir el DataFrame final
    
    # Bucle For para añadir cada operación al primer quirófano disponible
    for idx_op, new_op in datos.iterrows():
        op_asignada = False                                                    # Variable booleana que indica si la operación está asignada a un quirófano (se reinicializa en cada bucle for)
        i = 0                                                                  # índice del quirófano de lo cual empezar (se reinicializa en cada bucle for)
        
        while i < len(q_activos) and not op_asignada:
            q = q_activos[i]
            
            if len(K_dict[q]) == 0:                                            # Si el quirófano está vacío
                op_asignada = True                                               # -> asigna la operación
            else:                                                              # Si el quirófano NO está vacío
                ultima_operacion = K_dict[q][-1]                                 # -> obtiene el código de la última operación asignada
                t_fin_ultima = datos.loc[ultima_operacion, 'Hora fin']           # -> recupera la hora de fin del DataFrame
                if t_fin_ultima <= new_op['Hora inicio']:                        # Comprobar condición temporal (t_i_nueva > t_f_vieja)
                    op_asignada = True                                             # -> si es comprobada, asigna la operación
            
            if op_asignada:                                                    # Si la operación está asignada
                K_dict[q].append(idx_op)                                         # -> asigna la operación al quirófano
                asignaciones.append((idx_op, q))                                 # -> añade la asignación
            else:                                                              # Si la operación NO está asignada
                i += 1                                                           # -> pasar al siguiente quirófano
        
        if not op_asignada:                                                    # Si la operación aún NO está asignada
            q = quirofanos[len(q_activos)]                                       # -> abre un NUEVO quirófano
            K_dict[q] = [idx_op]                                                 # -> asigna la operación al quirófano
            q_activos.append(q)                                                  # -> actualización de los quirófanos abiertos
            asignaciones.append((idx_op, q))                                     # -> añade la asignación
    
    # Construcción de DataFrame
    K_df = pd.DataFrame(0, index = datos.index, columns = quirofanos)          # Creación del DataFrame
    for idx_op, q in asignaciones:
        K_df.loc[idx_op, q] = 1                                                # B_ik = 1, si la operación i está en la planificación del quirófano k
    
    # Coste de la planificacion
    C_i = {i : costes[i].mean() for i in datos.index}                          # Coste medio de cada operación
    C_k = {quirofano : round(sum(C_i[op] for op in operaciones), 2) 
           for quirofano, operaciones in K_dict.items()}                       # Suma los costes de las operaciones asignadas al quirófano

    # Devuelve el DataFrame y el coste de planificación
    return K_df, C_k         


# Datos -----------------------------------------------------------------------
costes = pd.read_excel('241204_costes.xlsx', index_col=0)                       # Importación costes desde archivos excel
datos = pd.read_excel('241204_datos_operaciones_programadas.xlsx', index_col=0) # Importación datos desde archivos excel

operaciones = ['Cardiología Pediátrica', 'Cirugía Cardíaca Pediátrica',
               'Cirugía Cardiovascular', 'Cirugía General y del Aparato Digestivo']
datos_filtrados = datos[datos['Especialidad quirúrgica'].isin(operaciones)]


# Generación de una planificación factible ------------------------------------
K, C_k = nueva_planificacion(datos_filtrados, costes)                          # K: Planificación;  C_k: Coste de la planificación


# Modelo del problema ---------------------------------------------------------
# Declaracion del modelo del problema
model_2 = lp.LpProblem(name = "Problema", sense = lp.LpMinimize)               # Problema de minimización de costes de asignación

# Declaracion variables de decisión
y = lp.LpVariable.dicts("y", [k for k in K.columns], cat = lp.LpBinary)        # Una variable para cada planificaciones de los quirófanos

# Declaracion de la función objetivo
model_2 += lp.lpSum(C_k[(k)]*y[(k)] for k in K.columns)                        # Costes de asignación

# Declaracion restricciones
for i in K.index:                                                              # Una restricción para cada operación
    model_2 += lp.lpSum(K.loc[i,k] * y[k] for k in K.columns) >= 1

# Tras la resolución e impresión de resultados
model_2.solve()
print('Estado del problema = ', lp.LpStatus[model_2.status])

for v in model_2.variables():
    print(v.name, '=', v.value())
    
print('Objective = ', lp.value(model_2.objective))