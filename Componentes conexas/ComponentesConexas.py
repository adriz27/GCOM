# -*- coding: utf-8 -*-
import random
import numpy as np
import matplotlib.pyplot as plt
import time

"""
Autor: Adrián Pérez

Parte 1 de Robert Monjo
"""

################################# PARTE 1 #####################################

#Generamos 1000 segmentos aleatorios, pero siempre serán los mismos

#Usaremos primero el concepto de coordenadas
X = []
Y = []

#Fijamos el modo aleatorio con una versión prefijada. NO MODIFICAR!!
random.seed(a=1, version=2)

#Generamos subconjuntos cuadrados del plano R2 para determinar los rangos de X e Y
xrango1 = random.sample(range(100, 1000), 200)
xrango2 = list(np.add(xrango1, random.sample(range(10, 230), 200)))
yrango1 = random.sample(range(100, 950), 200)
yrango2 = list(np.add(yrango1, random.sample(range(10, 275), 200)))
        
for j in range(len(xrango1)):
    for i in range(5):
        random.seed(a=i, version=2)
        xrandomlist = random.sample(range(xrango1[j], xrango2[j]), 4)
        yrandomlist = random.sample(range(yrango1[j], yrango2[j]), 4)
        X.append(xrandomlist[0:2])
        Y.append(yrandomlist[2:4])

#Representamos el Espacio topológico representado por los 1000 segmentos
        
for i in range(len(X)):
    plt.plot(X[i], Y[i], 'b')
plt.show()

###############################################################################

def intersecan(x1, y1, x2, y2):
    a1 = x1[1] - x1[0]
    b1 = x2[0] - x2[1]
    c1 = x2[0] - x1[0]
    a2 = y1[1] - y1[0]
    b2 = y2[0] - y2[1]
    c2 = y2[0] - y1[0]
    d = a1*b2-a2*b1
    if d == 0:  # Si son paralelos
        if a1*c2 != a2*c1:  # Si no están en la misma recta
            return False
        if x1[0] == x1[1]:
            if min(y1) > max(y2) or min(y2) > max(y1):
                return False
        else:
            if min(x1) > max(x2) or min(x2) > max(x1):
                return False
        return True
    t = (c1*b2-c2*b1)/d
    s = (a1*c2-a2*c1)/d
    return t >= 0 and t <= 1 and s >= 0 and s <= 1


def componentesConexas(_X, _Y):
    n = len(_X)
    comp = np.arange(n)
    for i in range(n):
        cambiado = False
        for j in range(i):
            if comp[i] != comp[j] and intersecan(_X[i], _Y[i], _X[j], _Y[j]):
                if cambiado:
                    for k in range(i+1):
                        if comp[k] == comp[i]:
                            comp[k] = comp[j]
                else:
                    comp[i] = comp[j]
                    cambiado = True
            
    for i in range(n):
        r = comp[i]/1000
        g = (comp[i] % 100)/100
        b = (comp[i] % 10) / 10
        plt.plot(X[i], Y[i], color = (r,g,b))
    plt.show()

    return comp


comp = componentesConexas(X, Y)
print("Número de componentes:", len(set(comp)))


def segmentosNuevos(minc, T, X, Y, comp):
    n = len(X)
    conectadas = set()
    nuevos = 0
    t = time.time()
    N = len(set(comp))
    
    while len(conectadas) < N:
        l1 = random.randint(0,3)
        l2 = l1
        while l1 == l2:
            l2 = random.randint(0,3)
        a1 = random.randint(0, 1250)
        a2 = random.randint(0, 1250)
        
        if l1 == 0:
            x1, y1 = 0, a1
        elif l1 == 1:
            x1, y1 = a1, 0
        elif l1 == 2:
            x1, y1 = a1, 1250
        else:
            x1, y1 = 1250, a1
            
        if l2 == 0:
            x2, y2 = 0, a2
        elif l2 == 1:
            x2, y2 = a2, 0
        elif l2 == 2:
            x2, y2 = a2, 1250
        else:
            x2, y2 = 1250, a2
    
        conecta = 0
        aux = []
        for i in range(n):
            if comp[i] not in conectadas and comp[i] not in aux and intersecan(X[i], Y[i], [x1,x2], [y1,y2]):
                conecta += 1
                if conecta < minc:
                    aux.append(comp[i])
                else:
                    conectadas.add(comp[i])
                    for c in aux:
                        conectadas.add(c)
        if conecta >= minc:
            t = time.time()
            nuevos += 1
            #print(len(conectadas))
            
        elif time.time() - t > T:
            if minc > 2:
                t = time.time()
                minc -= 1
            elif N - len(conectadas) <= 2:
                nuevos += 1 
                conectadas.add(-1)
                conectadas.add(-2)
    return nuevos
    
    
minc = 10
# Minimo de componentes conectadas inicialmente
T = 5
# Maximo tiempo
print("Segmentos para conectar todo:", segmentosNuevos(minc, T, X, Y, comp))
















