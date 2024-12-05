# -*- coding: utf-8 -*-
"""----------------------------------------------------------------------------
Metodos Cuantitativos Avanzados (MIO)

ENTREGA 3: Asignaciòn de quiròfanos en un hospital y generaciòn de columnas
Modelo 3 (Gen. columnas)

Grupo O
- KIANI HANIE
- PAEZ LEAL JAIRO EDUARDO
- PAJARES CAMACHO JAVIER
- TREVISAN FEDERICO
----------------------------------------------------------------------------""" 
#%%
''' PRIMER BLOQUE:
- Importación de librerías
- Definición de funciones
- Importación de datos
'''

# Importación de librerías ----------------------------------------------------
import pulp as lp                                                              # Para la resolución de problemas de PL 
import pandas as pd                                                            # Para trabajar con DataFrames

# Funciones -------------------------------------------------------------------

# Función para crear el conjunto de operaciones incompatibles de cada operación ---
def operaciones_incompatibles(datos):
    incompatibilidades = {op: set() for op in datos.index}
    for i, op_a in datos.iterrows():
        for j, op_b in datos.iterrows():
            if i != j:
                if (op_a["Hora inicio"] < op_b["Hora fin"]) and (op_b["Hora inicio"] < op_a["Hora fin"]):
                    incompatibilidades[i].add(j)
                    incompatibilidades[j].add(i)
    return incompatibilidades


# Función para la generacion de una prima planificación factible --------------
def primera_planificacion(datos, costes):
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
    
    # Coste de la planificación
    C_i = {i : costes[i].mean() for i in datos.index}                          # Coste medio de cada operación
    C_k = {quirofano : round(sum(C_i[op] for op in operaciones), 2) 
           for quirofano, operaciones in K_dict.items()}                       # Suma los costes de las operaciones asignadas al quirófano

    # Devuelve el DataFrame y el coste de planificación
    return K_df, C_k


# Problema de minimización de quirófanos (version 'relajada' con variables continuas) ---
def maestro_relajado(planificacion):
    
    # Declaracion del modelo del problema
    model_master = lp.LpProblem(name = "Problema", sense = lp.LpMinimize)      # Problema de minimización

    # Declaracion variables de decisión
    x = lp.LpVariable.dicts("x", [k for k in planificacion.columns], lowBound=0) # Una variable para cada quirófano

    # Declaracion de la función objetivo
    model_master += lp.lpSum(x[k] for k in planificacion.columns)              # Suma de los quirófanos abiertos

    # Declaracion restricciones
    for i in planificacion.index:                                              # Una restricción para cada operación
        model_master += lp.lpSum(planificacion.loc[i,k] * x[k] for k in planificacion.columns) >= 1 # Cada operación debe asignarse al menos a un quirófano
    
    # Tras la resolución
    model_master.solve()
    shadow_prices = {}
    for op, restriccion in zip(K.index, model_master.constraints.values()):    # Guardar precios sombra 
        shadow_prices[op] = restriccion.pi
    
    # Devuelve los precios sombra y la función objetivo
    return shadow_prices, lp.value(model_master.objective)


# Problema de minimización de quirófanos (version con variables binarias) -----
def maestro_entero(planificacion):
    
    # Declaracion del modelo del problema
    model_entero = lp.LpProblem(name = "Problema", sense = lp.LpMinimize)      # Problema de minimización

    # Declaracion variables de decisión
    x = lp.LpVariable.dicts("x", [k for k in planificacion.columns], cat = lp.LpBinary) # Una variable para cada quirófano (x[k]=1 si el quirófano k está abierto)

    # Declaracion de la función objetivo
    model_entero += lp.lpSum(x[k] for k in planificacion.columns)              # Suma de los quirófanos abiertos

    # Declaracion restricciones
    for i in planificacion.index:                                              # Una restricción para cada operación
        model_entero += lp.lpSum(planificacion.loc[i,k] * x[k] for k in planificacion.columns) >= 1 # Cada operación debe asignarse al menos a un quirófano
    
    # Tras la resolución
    model_entero.solve()
    
    # Devuelve las variables y la función objetivo
    return x, lp.value(model_entero.objective)


# Función que genera la nueva columna que se añadirá a la planificación -------
def generacion_columnas(shadow_prices, datos, incompatibilidades):
    
    # Declaracion del modelo del problema
    model_generacion = lp.LpProblem(name = "Problema", sense = lp.LpMaximize)  # Problema de maximización

    # Declarar variables de decisión
    y = lp.LpVariable.dicts("y", [i for i in datos.index], cat = lp.LpBinary)  # Una variable para cada operación (y[1]=1 si la operación i está asignada a esta planificación)

    # Declaracion de la función objetivo
    model_generacion += lp.lpSum(shadow_prices[i] * y[i] for i in datos.index) # Maximización de las operaciones asignadas

    # Declaracion restricciones
    for i in datos.index:                                                      # Una restricción para cada operación 
        for j in incompatibilidades[i]:
            model_generacion += y[i] + y[j] <= 1                               # Verifico que cada operación no entra en conflicto con sus incompatibles
    
    # Tras la resolución
    model_generacion.solve()
    
    # Devuelve las variables y la función objetivo
    return y, lp.value(model_generacion.objective)


# Datos -----------------------------------------------------------------------
costes = pd.read_excel('241204_costes.xlsx', index_col=0)                       # Importación costes desde archivos excel
datos = pd.read_excel('241204_datos_operaciones_programadas.xlsx', index_col=0) # Importación datos desde archivos excel

# Calcular incompatibilidades basadas en horarios
incompatibilidades = operaciones_incompatibles(datos)

#%%
''' SEGUNDO BLOQUE:
- Creación de una planificación inicial factible
- Primera ejecución de los problemas «maestro» y «generación de columnas»
- Bucle WHILE para mejorar la solución
'''

# Crear una planificación inicial ---------------------------------------------
K, C_k = primera_planificacion(datos, costes)
#K = pd.DataFrame({j+1: [1 if i == j else 0 for i in range(len(datos.index))] for j in range(len(datos.index))}, index=datos.index)

# Primera ejecución de los problemas ------------------------------------------
print("Reducción del quirofano necesario ...")
shadow_prices_maestro, quirofanos_activos = maestro_relajado(K)                                # Master problem (recibe los precios sombra y la función objetivo del problema maestro)
nueva_columna, control = generacion_columnas(shadow_prices_maestro, datos, incompatibilidades) # Generación columnas (recibe las variables y la función objetivo del problema de generación de columnas)
print("\nQuirofanos activos -> ", quirofanos_activos, "\nCondición de salida -> ", control)

# Bucle WHILE -----------------------------------------------------------------
# -> la condición de salida verifica que realmente se está mejorando la función objetivo del problema maestro
while control > 1:
    
    # La nueva columna se añade a la planificación
    K[f"Quirófano {len(K.columns)+1}"] = [nueva_columna[idx].varValue if nueva_columna[idx].varValue is not None else 0 for idx in K.index]
    
    # Ejecución de los problemas
    shadow_prices_maestro, quirofanos_activos = maestro_relajado(K)                                 # Master problem 
    nueva_columna, control = generacion_columnas(shadow_prices_maestro, datos, incompatibilidades)  # Generación columnas
    print("\nQuirofanos activos -> ", quirofanos_activos, "\nCondición de salida -> ", control)

print("Nueva planificación obtenida:\n\n", K)
#%%    
''' TERCER  BLOQUE:
- Buscar la solución óptima mediante el problema maestro entero
'''
quirofanos, n_quirofanos = maestro_entero(K)

print(f"\n\nEs necesario activar {n_quirofanos} quirofanos para llevar a cabo la planificación diaria")

plan_final = {}
for k in K.columns:
    if quirofanos[k].varValue:
        plan_final[k] = K.index[K[k] == 1].tolist()

print("\n\nPlanificación final: ")
for k in plan_final.keys():
    print(k, " -> ", plan_final[k])


