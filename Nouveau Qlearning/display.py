# -*- coding: utf-8 -*-


import numpy as np
import tkinter as tk


class Displayer():
    def __init__(self):
        self.WIDTH = 500
        self.HEIGHT = 500
        
    def _make_square_box(self, i, j, mouse=False):
        """
        Make square bounding box from indices.
        If mouse is True, the bounding box produced is smaller.
        """
        
        if mouse is True:
            return [self.square_size * elem for elem in ((i+0.2), (j+0.2), (i+0.8), (j+0.8))]
        else: 
            return [self.square_size * elem for elem in (i, j, i+1, j+1)]
        
        
    def create_labyrinth(self, matrix_labyrinth, mouse_initial_indices):
        
        self.matrix = matrix_labyrinth
        self.square_size = self.WIDTH // (np.max(self.matrix.shape)+1) 
        
        self.window = tk.Tk()
        self.window.title('labyrinth')
        
        
        self.window.geometry('%sx%s'%(self.WIDTH,self.HEIGHT))
        self.main_canva = tk.Canvas(self.window, width=self.WIDTH, height=self.HEIGHT-30, bg='white')
        self.main_canva.pack()
        self.main_canva.focus_set()
        
        for i in range(np.shape(self.matrix)[0]):
            for j in range(np.shape(self.matrix)[1]):
                
                square_box = self._make_square_box(i,j)
                
                if self.matrix[i,j] == 0:
                    self.main_canva.create_rectangle(*square_box, fill="black")
                
                if self.matrix[i,j] == 1:
                    self.main_canva.create_rectangle(*square_box, fill="white")
                            
                if self.matrix[i,j] == 2:
                    self.main_canva.create_rectangle(*square_box, fill="green")
            
                if self.matrix[i,j] == 3:
                    self.main_canva.create_rectangle(*square_box, fill="red")
                
                if self.matrix[i,j] == 4:
                    self.main_canva.create_rectangle(*square_box, fill="blue")


        self.mouse = self.main_canva.create_rectangle(
            self._make_square_box(*mouse_initial_indices, mouse=True),
            fill="grey"
        )

if __name__=="__main__":
    
    L = np.array(  [[0,0,0,0,0,0,0,0,0,0],
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
                    ]).T #labyrinth utilisé (0=mur, 1=vide, 2= arrivée, 3=électricité, 4=eau)
    
    displayer = Displayer()
    displayer.create_labyrinth(L, mouse_initial_indices)
    displayer.window.mainloop()
    
    
