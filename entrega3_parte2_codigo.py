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
import itertools

# Funciones -------------------------------------------------------------------

def verify_time(datos, matrix):
    """
    Verifica la compatibilità della pianificazione delle operazioni nelle sale operatorie.
    
    Args:
        A (pd.DataFrame): Matrice nxm che indica l'assegnazione delle operazioni alle sale operatorie.
        datos (pd.DataFrame): DataFrame con gli orari di inizio e fine delle operazioni.
                              
    Returns:
        bool: True se la pianificazione è fattibile, False altrimenti.
    """
    
    for sala in matrix.index:
        # Trova le operazioni assegnate alla sala corrente
        operazioni_assegnate = matrix.columns[matrix.loc[sala] == 1]
        
        # Controlla la compatibilità degli orari per la sala
        for i in range(len(operazioni_assegnate) - 1):
            op_corrente = operazioni_assegnate[i]
            op_successiva = operazioni_assegnate[i + 1]
            
            # Ottieni gli orari di fine e inizio delle operazioni
            h_fin_corrente = datos.loc[op_corrente, 'Hora fin']
            h_inicio_successiva = datos.loc[op_successiva, 'Hora inicio']
            
            # Verifica la condizione
            if h_fin_corrente >= h_inicio_successiva:
                return False  # Pianificazione non valida
    
    return True  # Pianificazione valida

def generate_direct_matrices(costes, datos):
    """
    Genera direttamente tutte le matrici che soddisfano la regola:
    La somma degli elementi di ogni colonna è uguale a 1.
    
    Args:
    costes (df): Dataframe con i costi.
    datos  (df): Dataframe con le operazioni.
    
    Yields:
    pd.DataFrame: Una matrice valida.
    """
    n = len(costes.index)   # Numero di righe
    m = len(datos.index)    # Numero di colonne
 
    # Per ogni colonna, seleziona una sola riga in cui mettere `True`
    row_indices = range(n)
    for col_selections in itertools.product(row_indices, repeat=m):
        # Crea una matrice n x m inizialmente tutta False
        matrix = np.zeros((n, m), dtype=bool)
        for col, row in enumerate(col_selections):
            matrix[row, col] = True  # Imposta la cella selezionata a True
            matrix_df = pd.DataFrame(matrix) # Converte la matrice in un df
            matrix_df.index = [costes.index]
            matrix_df.columns = [datos.index]
            #Verifica che la pianificazione generata rispetti gli orari
            if verify_time(datos, matrix_df):
                yield matrix_df

# Datos -----------------------------------------------------------------------
datos = pd.read_excel('241204_datos_operaciones_programadas.xlsx', index_col=0) # Importación datos desde archivos excel
costes = pd.read_excel('241204_costes.xlsx', index_col=0)                       # Importación costes desde archivos excel

operaciones = ['Cardiología Pediátrica', 'Cirugía Cardíaca Pediátrica', 'Cirugía Cardiovascular', 'Cirugía General y del Aparato Digestivo']
datos_filtrados = datos[datos['Especialidad quirúrgica'].isin(operaciones)]

# Generazione delle pianificazioni possibili
K = generate_direct_matrices(costes, datos_filtrados)

# Stampa alcune matrici valide
for idx, df in zip(range(5), K):  # Mostra solo le prime 5
    print(f"Matrice valida {idx + 1}:\n{df}\n")

# Costo medio di ogni operazione
C_i = {i : costes[i].mean() for i in datos_filtrados.index}

# Costo di ogni pianificazione
C_k = {idx : k for idx,k in enumerate(K, start=1)}

inicio = datos_filtrados.iloc[0]['Hora inicio']
fin = datos_filtrados.iloc[0]['Hora fin']

inicio > fin



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