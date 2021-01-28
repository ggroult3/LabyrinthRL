# -*- coding: utf-8 -*-

import numpy as np
import tkinter as tk

#test

class affichage():
    def __init__(self):
        self.WIDTH = 1500
        self.HEIGHT = 1500
        self.labyrinthe = np.array([[0,0,0,0,0,0,0,0,0,0],
                                    [0,1,1,3,0,0,1,1,1,0],
                                    [0,0,1,1,1,1,1,0,4,0],
                                    [0,1,1,0,1,0,0,0,1,0],
                                    [0,1,0,0,1,1,1,0,1,0],
                                    [0,1,0,0,0,0,3,0,1,0],
                                    [0,1,0,1,1,0,1,0,1,0],
                                    [0,1,1,1,0,0,1,1,1,0],
                                    [0,0,0,1,0,0,0,1,0,0],
                                    [0,4,1,1,1,1,1,1,2,0],
                                    [0,0,0,0,0,0,0,0,0,0]
                                    ]).T #labyrinthe utilisé (0=mur, 1=vide, 2= arrivée, 3=électricité, 4=eau)
        self.taille_case = 80
        
    def creer_lab(self):
        self.fenetre = tk.Tk()
        self.fenetre.title('Labyrinthe')
        self.fenetre.geometry('%sx%s'%(self.WIDTH,self.HEIGHT))
        self.canvaPrincipale = tk.Canvas(self.fenetre, width=self.WIDTH, height=self.HEIGHT-30, bg='white')
        self.canvaPrincipale.pack()
        self.canvaPrincipale.focus_set()
        L=self.labyrinthe
        souris_placee = False
        
        for i in range(np.shape(L)[0]):
            for j in range(np.shape(L)[1]):
                if L[i,j] == 0:
                    self.canvaPrincipale.create_rectangle(self.taille_case*i,self.taille_case*j,self.taille_case*(i+1),self.taille_case*(j+1), fill="black")
                
                if L[i,j] == 1:
                    self.canvaPrincipale.create_rectangle(self.taille_case*i,self.taille_case*j,self.taille_case*(i+1),self.taille_case*(j+1), fill="white")
                    if not souris_placee :
                        self.souris = self.canvaPrincipale.create_rectangle(self.taille_case*(i+0.2),self.taille_case*(j+0.2),self.taille_case*(i+0.8),self.taille_case*(j+0.8), fill="grey")
                        souris_placee = True #Souris placée sur la case vide le plus en haut à gauche.
                            
                if L[i,j] == 2:
                    self.canvaPrincipale.create_rectangle(self.taille_case*i,self.taille_case*j,self.taille_case*(i+1),self.taille_case*(j+1), fill="green")
            
                if L[i,j] == 3:
                    self.canvaPrincipale.create_rectangle(self.taille_case*i,self.taille_case*j,self.taille_case*(i+1),self.taille_case*(j+1), fill="red")
                
                if L[i,j] == 4:
                    self.canvaPrincipale.create_rectangle(self.taille_case*i,self.taille_case*j,self.taille_case*(i+1),self.taille_case*(j+1), fill="blue")

 
lab = affichage()
lab.creer_lab()
lab.fenetre.mainloop()

