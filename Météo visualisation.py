import tkinter as tk
import tkinter.ttk as ttk
import modele_meteo as mm
import numpy as np
from geopy.geocoders import Nominatim


class visualitation:

    def __init__(self):
        self.fen = tk.Tk()
        self.fen.title("Prédiction Météo")
        self.text = tk.Text(self.fen, height=19)
        self.text.insert(tk.INSERT, "Que voulez-vous faire ?")
        self.text.pack(side='top')
        self.a1 = tk.Button(self.fen, text='Charger un modèle existant', command=self.charge_modele)
        self.a1.pack(side='top')
        self.a2 = tk.Button(self.fen, text='Entrainer un nouveau modèle', command=self.choix_parametres)
        self.a2.pack(side='top')
        self.modele = mm.prediction_meteo(100, 50)
        self.fen.mainloop()

    # II.fonction utile
    def choix_parametres(self):
        self.a1.destroy()
        self.text.destroy()
        self.a2.destroy()
        self.choix_station()


    def charge_modele(self):
        self.a1.destroy()
        self.text.destroy()
        self.a2.destroy()
        print('top')

    def selection_data(self):
        self.s1 = tk.Button(self.fen, text="Charger les données", command=self.creer_curseur)
        self.s1.pack(side='top')

    def creer_curseur(self):
        self.v1 = tk.IntVar()
        self.b1 = tk.Checkbutton(self.fen, text='2016', variable=self.v1, width=35)
        self.b1.pack(side='left')
        self.v2 = tk.IntVar()
        self.b2 = tk.Checkbutton(self.fen, text='2017', variable=self.v2, width=35)
        self.b2.pack(side='left')
        self.v3 = tk.IntVar()
        self.b3 = tk.Checkbutton(self.fen, text='2018', variable=self.v3, width=35)
        self.b3.pack(side='left')
        self.b4 = tk.Button(self.fen, text='Valider', command=self.retour)
        self.b4.pack(side='left')

    def retour(self):
        self.b1.destroy()
        self.b2.destroy()
        self.b3.destroy()
        self.b4.destroy()
        annee = []
        if self.v1.get()==1:
            annee.append(2016)
        elif self.v2.get()==1:
            annee.append(2017)
        elif self.v3.get()==1:
            annee.append(2018)
        self.modele.chargement(annee)


    def distance(self,P1,P2):
        return ((P1[0] - P2[0]) ** 2 + (P1[1] - P2[1]) ** 2) ** (1 / 2)

    def plus_proche(self):
        nom_coord="D:\TIPE\Python\coord station.npz"
        coord=np.load(nom_coord)
        num,lat,lon=[],[],[]
        for i in range(len(coord['arr_0'])):
            num.append(coord['arr_0'][i][0])
            lat.append(coord['arr_0'][i][1])
            lon.append(coord['arr_0'][i][2])
        self.mini = 0
        m = 9999999999
        for i in range(len(lat)):
            P=[lat[i],lon[i]]
            dist=self.distance(P, [self.y1, self.x1])
            if dist<m:
                self.mini=i
                m=dist
        geolocator = Nominatim(user_agent="geoapiExercises")
        return 'Ville : '+str(geolocator.reverse(str(lat[self.mini])+","+str(lon[self.mini])))+\
               '\nLatitude : '+str(lat[self.mini])+' / Longitude : '+str(lon[self.mini])

    def clique_ville(self,event):
        x0, y0 = event.x, event.y
        self.text.destroy()
        if y0>531:
            self.text = tk.Text(self.fen, height=4, width=60)
            self.text.insert(tk.INSERT, "En dehors de la carte !!!")
            self.text.pack(side='bottom')
        else:
            self.text = tk.Text(self.fen, height=4,width=60)
            lllat = 46.25  # latitude basse gauche
            urlat = 51.896  # latitude haute droite
            lllon = -5.842  # longitude basse gauche
            urlon = 2  # longitude haute droite
            self.x1=x0/498*(urlon-lllon)+lllon
            self.y1=(531-y0)/531*(urlat-lllat)+lllat
            self.proche=self.plus_proche()
            self.text.insert(tk.INSERT, self.proche)
            self.text.pack(side='bottom')


    def choix_station(self):
        self.canvas = tk.Canvas(self.fen, bg="light gray", height=531, width=498)
        self.text = tk.Text(self.fen, height=4, width=60)
        self.text.pack(side='bottom')
        img = tk.PhotoImage(file="CarteNW.png")
        self.canvas.create_image(0,0,anchor=tk.NW,image=img)
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)
        self.fen.bind("<Button-1>",self.clique_ville)
        self.b4 = tk.Button(self.fen, text='Valider', command=self.creer_curseur)
        self.b4.pack(side='bottom')
        self.fen.mainloop()#trés important

 

visualitation()
