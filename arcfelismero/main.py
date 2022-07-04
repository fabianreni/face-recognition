import numpy as np
import cv2
import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pandas import DataFrame
import os
plt.style.use('ggplot')
import tkinter as tk
from tkinter import filedialog
import Eigenface
import Fisherface
import Lbph

def ablak():
    def donothing():
        pass

    def keptesztbezarCallback():
        root.destroy()
    root= tk.Tk()
    root.title("Arcfelismerő")
    root.protocol('WM_DELETE_WINDOW',donothing)

    canvas1 = tk.Canvas(root, width =250, height = 300)
    
    canvas1.pack()

    button1= tk.Button(text='LBPH teszt!', command=Lbph.lbph)

    button5= tk.Button(text='LBPH tanitas!',  command=Lbph.train)
    canvas1.create_window(175, 40, window=button1)
    canvas1.create_window(75, 20, window=button5)

    button2= tk.Button(text='Eigenface  teszt!', command=Eigenface.eigenFace)
    button8= tk.Button(text='Eigenface tanitas!', command=Eigenface.train)
    canvas1.create_window(175, 120, window=button2)
    canvas1.create_window(75, 100, window=button8)
 
    button3= tk.Button(text='Fisherface teszt!', command=Fisherface.fisherFace)
    button6= tk.Button(text='Fisherface tanitas!', command=Fisherface.train)
    canvas1.create_window(175, 200, window=button3)  
    canvas1.create_window(75, 180, window=button6) 

    button = tk.Button(text='Befejez', command=keptesztbezarCallback)
    canvas1.create_window(135,250, window=button)

    root.mainloop()
ablak()