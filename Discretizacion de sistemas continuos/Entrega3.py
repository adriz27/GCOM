# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d

def F(q):
    ddq = -8/3*q*(q**2-1/2)
    return ddq

def orb(n,q0,dq0,F, args=None, d=0.001):
    #q = [0.0]*(n+1)
    q = np.empty([n+1])
    q[0] = q0
    q[1] = q0 + dq0*d
    for i in np.arange(2,n+1):
        args = q[i-2]
        q[i] = - q[i-2] + d**2*F(args) + 2*q[i-1]
    return q #np.array(q),

#q = variable de posición, dq0 = \dot{q}(0) = valor inicial de la derivada
#d = granularidad del parámetro temporal
def deriv(q,dq0,d):
   #dq = np.empty([len(q)])
   dq = (q[1:len(q)]-q[0:(len(q)-1)])/d
   dq = np.insert(dq,0,dq0) #dq = np.concatenate(([dq0],dq))
   return dq


#gráfico del oscilador no lineal
q0 = 0.
dq0 = 1.
fig, ax = plt.subplots(figsize=(12,5))
#plt.ylim(0, 1)  
plt.rcParams["legend.markerscale"] = 6
ax.set_xlabel("t = n $\delta$", fontsize=12)
ax.set_ylabel("q(t)", fontsize=12)
iseq = np.array([3,4])
horiz = 32
for i in iseq:
    d = 1./10**i
    n = int(horiz/d)
    t = np.arange(n+1)*d
    q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
    plt.plot(t, q, 'ro', markersize=0.5/i,label='$\delta$ ='+
             str(np.around(d,4)),c=plt.get_cmap("winter")(i/np.max(iseq)))
    ax.legend(loc=3, frameon=False, fontsize=12)
    
d = 1./10**4
n = int(horiz/d)
t = np.arange(n+1)*d
q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
dq = deriv(q,dq0=dq0,d=d)
p = dq/2

#grafica derivada de q(t)
fig, ax = plt.subplots(figsize=(12,5))
#plt.ylim(-1.5, 1.5)  
plt.rcParams["legend.markerscale"] = 6
ax.set_xlabel("t = n $\delta$", fontsize=12)
ax.set_ylabel("dq(t)", fontsize=12)
plt.plot(t, dq, '-')

#Ejemplo de diagrama de fases (q, p) para una órbita completa
fig, ax = plt.subplots(figsize=(5,5))
#plt.xlim(-1.1, 1.1)  
#plt.ylim(-1, 1) 
plt.rcParams["legend.markerscale"] = 6
ax.set_xlabel("q(t)", fontsize=12)
ax.set_ylabel("p(t)", fontsize=12)
plt.plot(q, p, '-')
plt.show()


#################################################################    
#  ESPACIO FÁSICO
################################################################# 

def simplectica(q0,dq0,F,col=0,d = 10**(-4),n = int(16/d),marker='-'): 
    q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
    dq = deriv(q,dq0=dq0,d=d)
    p = dq/2
    plt.plot(q, p, marker,c=plt.get_cmap("winter")(col))


Horiz = 12
d = 10**(-4)

fig = plt.figure(figsize=(8,5))
fig.subplots_adjust(hspace=0.4, wspace=0.2)
ax = fig.add_subplot(1,1, 1)
#Condiciones iniciales:
seq_q0 = np.linspace(0.,1.,num=10)
seq_dq0 = np.linspace(0.,2.,num=10)
for i in range(len(seq_q0)):
    for j in range(len(seq_dq0)):
        q0 = seq_q0[i]
        dq0 = seq_dq0[j]
        col = (1+i+j*(len(seq_q0)))/(len(seq_q0)*len(seq_dq0))
        #ax = fig.add_subplot(len(seq_q0), len(seq_dq0), 1+i+j*(len(seq_q0)))
        simplectica(q0=q0,dq0=dq0,F=F,col=col,marker='-',d= 10**(-4),n = int(Horiz/d))
ax.set_xlabel("q(t)", fontsize=12)
ax.set_ylabel("p(t)", fontsize=12)
#fig.savefig('Simplectic.png', dpi=250)
plt.show()


#Ejemplo de diagrama de fases (q, p) para un tiempo determinado

def evoluciondf (t, d, N):
    ax = fig.add_subplot(1,1, 1)
    seq_q0 = np.linspace(0,1,num=N)
    seq_dq0 = np.linspace(0,2,num=N)
    q2 = np.array([])
    p2 = np.array([])
    q3 = np.array([])
    p3 = np.array([])
    q4 = np.array([])
    p4 = np.array([])
    for i in range(N):
        for j in range(N):
            q0 = seq_q0[i]
            dq0 = seq_dq0[j]
            n = int(t/d)
            q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
            dq = deriv(q,dq0=dq0,d=d)
            p = dq/2
            q2 = np.append(q2,q[-1])
            p2 = np.append(p2,p[-1])
            if(j == 0):
                q3 = np.append(q3,q[-1])
                p3 = np.append(p3,p[-1])
            if(i == N-1):
                q4 = np.append(q4,q[-1])
                p4 = np.append(p4,p[-1])

            plt.xlim(-2.2, 2.2)
            plt.ylim(-1.2, 1.2)
            plt.rcParams["legend.markerscale"] = 6
            ax.set_xlabel("q(t)", fontsize=12)
            ax.set_ylabel("p(t)", fontsize=12)
            plt.plot(q[-1], p[-1], marker="o", markersize= 2, 
                     markeredgecolor="red",markerfacecolor="red")

    plt.show()

    X = np.array([q2,p2]).T
    hull = ConvexHull(X)
    #convex_hull_plot_2d(hull)

    Y = np.array([q3,p3]).T
    Y_hull = ConvexHull(Y)
    #convex_hull_plot_2d(Y_hull)
    
    Z = np.array([q4,p4]).T
    Z_hull = ConvexHull(Z)
    #convex_hull_plot_2d(Z_hull)

    print("Perímetro:", hull.area)
    print("Área:", hull.volume- Y_hull.volume- Z_hull.volume)        
    #print("Vértices:", X[hull.vertices])

evoluciondf(0.25, 10**(-4), 20)
evoluciondf(0.25, 10**(-5), 40)
evoluciondf(0.001, 10**(-4), 20)
evoluciondf(0.5, 10**(-4), 20)


from matplotlib import animation

def animate(t):
    
    ax = plt.axes()
    ax.clear()
    d = 10**(-4)
    if (t == 0):
        t += 0.001
    n = int(t/d)
    N = 100
    seq_q0 = np.linspace(0,1,num=N)
    seq_dq0 = np.linspace(0,2,num=N)
    
    for i in range(N):
        for j in range(N):    
            if i == 0 or i == N-1 or j == 0 or j == N-1:
                q0 = seq_q0[i]
                dq0 = seq_dq0[j]    
                q = orb(n,q0=q0,dq0=dq0,F=F,d=d)
                dq = deriv(q,dq0=dq0,d=d)
                p = dq/2
                plt.xlim(-2.2, 2.2)
                plt.ylim(-1.2, 1.2)
                plt.rcParams["legend.markerscale"] = 6
                ax.set_xlabel("q(t)", fontsize=12)
                ax.set_ylabel("p(t)", fontsize=12)
                ax.plot(q[-1], p[-1], marker="o", markersize= 1, 
                        markeredgecolor="red",markerfacecolor="red")
    return ax,

fig = plt.figure(figsize=(6,6))
ani = animation.FuncAnimation(fig, func=animate, 
                              frames=np.arange(0,5,0.1), interval=100)
ani.save("animation.gif", fps = 8)