import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

"""
Autor: Adrián Pérez
"""     

def complejidadCod(muestra):
    #### Contamos cuantos caracteres hay
    tab = Counter(muestra)
    weights = np.array(list(tab.values()))
    prob = weights/float(np.sum(weights))
    prob.sort()
    
    # Se crean los vectores con la equidistribución y la curva de Lorentz
    n = len(prob)
    prob_xy = np.linspace(0.,1.,n+1)
    prob_acum = []
    prob_acum.append(0)
    for i in range(n):
        prob_acum.append(prob_acum[i] + prob[i])
    
    # Se representan gráficamente 
    plt.plot(prob)
    plt.title("Función de probabilidad")
    plt.show()
    plt.plot(prob_acum)
    plt.plot(prob_xy)
    plt.legend(("Curva de Lorentz", "Equidistribución"))
    plt.show()
    
    # Calculo correspondiente a la diversidad e indice de Gini
    A = 0.5 # Parte correspondiente a los triangulos superiores de los trapecios
    D = 0
    for i in range(n):
        D += prob[i]**2
        A += prob_acum[i]
    A /= n
    D = 1./D
    GI = 1 - 2*A
    
    print("Area bajo la curva: ", A)
    print("Indice de Gini: ", GI)
    print("Diversidad: ", D)
    print("Numero de elementos: ", n)
    
muestra1 = [0, 1, 0, 0, 0, 2, 1, 1, 0, 2, 0, 1, 2, 0, 2, 0, 1, 1]

print("Muestra de ejemplo\n")
complejidadCod(muestra1)


#### Vamos al directorio de trabajo
os.getcwd()

with open('GCOM2023_pract1_auxiliar_eng.txt', 'r',encoding="utf8") as file:
      en = file.read()
      
print("-----------------\nS_English\n")
complejidadCod(en)