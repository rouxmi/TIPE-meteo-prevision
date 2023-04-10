# I.importation


# repésentation des résultats
import matplotlib.pyplot as plt
# algèbre linéaire
import numpy as np
# transforamtion des données et importation depuis le csv
import pandas as pd
import seaborn as sns
# Machine learning
import tensorflow as tf


class prediction_meteo:

    def __init__(self, point_hist, point_predis):
        self.point_hist = point_hist
        self.point_predis = point_predis


    # Normalisation des valeurs entre 0 et 1
    def normaliser(self, dataset, parametre, parametre_unique=False):
        if parametre_unique:
            dataNorm = dataset
            dataNorm[parametre] = ((dataset[parametre] - dataset[parametre].min()) / (
                    dataset[parametre].max() - dataset[parametre].min()))
            return dataNorm
        else:
            dataNorm = ((dataset - dataset.min()) / (dataset.max() - dataset.min()))
            return dataNorm

    def denormaliser(self, dataset, dataset_origine, parametre, parametre_unique=False):
        if parametre_unique:
            dataNorm = dataset
            dataNorm[parametre] = (
                    (dataset[parametre]) * (dataset_origine[parametre].max() - dataset_origine[parametre].min()) + (
                dataset_origine[parametre].min()))
            return dataNorm
        else:
            dataNorm = ((dataset) * (dataset_origine.max() - dataset_origine.min()) + (dataset_origine.min()))
            return dataNorm

    def segment(self, dataset, variable, intervale=5000, futur=0):
        data = []
        labels = []
        for i in range(len(dataset)):
            fin_index = i + intervale
            futur_index = i + intervale + futur
            if futur_index >= len(dataset):
                break
            data.append(dataset[variable][i:fin_index])
            labels.append(dataset[variable][fin_index:futur_index])
        return np.array(data), np.array(labels)

    def temp_int(self, longeur):
        return list(range(-longeur, 0))

    def plot_multi_etape(self, historique, vrai_futur, prediction):
        plt.figure(figsize=(12, 6))
        num_passe = self.temp_int(len(historique))
        num_futur = len(vrai_futur)
        plt.plot(num_passe, np.array(historique[:, 0]), label='Passé donné')
        plt.plot(np.arange(num_futur), np.array(vrai_futur), label='Véritable futur')
        if prediction.any():
            plt.plot(np.arange(num_futur), np.array(prediction), 'ro', label='Futur prédis')
        plt.legend(loc='upper left')
        plt.show()

    # III.Transformation des données

    def chargement(self, annee):
        # importation de notre dataset sous panda
        if 2016 in annee:
            self.df2016 = pd.read_csv('D:\TIPE\Python\Data TIPE\ground station\\2016nord-ouest.csv')
        elif 2017 in annee:
            self.df2017 = pd.read_csv('D:\TIPE\Python\Data TIPE\ground station\\2017nord-ouest.csv')
        elif 2018 in annee:
            self.df2018 = pd.read_csv('D:\TIPE\Python\Data TIPE\ground station\\2018nord-ouest.csv')

    def index(self, station, annee=None):
        if annee is None:
            annee = list([2016])
        if annee != [] and (2016 in annee or 2016 in annee or 2018 in annee):
            if 2016 in annee:
                self.data = self.df2016[(self.df2016['number_sta'] == station)]
                if 2017 in annee:
                    self.data = self.data.append(self.df2017[(self.df2017['number_sta'] == station)], ignore_index=True)
                if 2018 in annee:
                    self.data = self.data.append(self.df2018[(self.df2018['number_sta'] == station)], ignore_index=True)
            elif 2017 in annee:
                self.data = self.df2017[(self.df2017['number_sta'] == station)]
                if 2018 in annee:
                    self.data = self.data.append(self.df2018[(self.df2018['number_sta'] == station)], ignore_index=True)
            else:
                self.data = self.df2016[(self.df2016['number_sta'] == station)]

            self.data['date'] = pd.to_datetime(self.data['date'], format='%Y%m%d %H:%M')
            self.data.set_index('date', inplace=True)
            self.data['td'] = self.data['td'].interpolate('linear')
            self.data['precip'] = self.data['precip'].interpolate('linear')
            self.data['hu'] = self.data['hu'].interpolate('linear')
            self.data['ff'] = self.data['ff'].interpolate('linear')
            self.data = self.data.drop(['number_sta', 'lat', 'lon', 'height_sta'], axis=1)

    def transformation_donnee(self, station_meteo, test_station_meteo, annee):
        # création de nos données
        meteo_non_norm = self.index(station_meteo, annee)
        meteo_test_non_norm = self.index(test_station_meteo, annee)

        # normalisation pour accélerer le calcul
        self.meteo = self.normaliser(meteo_non_norm, 'td', parametre_unique=False)
        self.meteo_test = self.normaliser(meteo_test_non_norm, 'td', parametre_unique=False)

        # on prend la moyenne sur des périodes de 720 messures(5 min) = 12 heures
        self.meteo_cp = self.meteo.resample('720T').mean()
        self.meteo_test_cp = self.meteo_test.resample('720T').mean()

        # on considère la température constante si on a pas de données :
        self.meteo_cp = self.meteo_cp.fillna(method='bfill')
        self.meteo_test_cp = self.meteo_test_cp.fillna(method='bfill')

    # IV.Création du modèle

    # Selection du nombre de point donné et ceux à prédir

    # Prépare des sacs de données suivant la sélection et imprime la Dimension des données (permet de voir si le volume de données est trop faible ou trop grand)
    def segmentation(self, dataset, parametre):
        X, Y = self.segment(dataset, parametre, intervale=self.point_hist, futur=self.point_predis)
        X = X.reshape(X.shape[0], self.point_hist, 1)
        Y = Y.reshape(Y.shape[0], self.point_predis, 1)
        return X, Y

    def creation_segement(self, data, data_test, parametre):
        X_entrainement, Y_entrainement = self.segmentation(data, parametre)
        X_test, Y_test = self.segmentation(data_test, parametre)
        return X_entrainement, Y_entrainement, X_test, Y_test

    # Nombre d'époque d'entrainement(résultat satisfaisant vers 200)

    # modèle du réseaux de neurones(4 rangées (100,100,50,50) dont la première LSTM)
    def cree_modele(self, X_entrainement):
        self.modele_lstm = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(self.point_hist, input_shape=X_entrainement.shape[-2:]),
            tf.keras.layers.Dense(100, activation='elu'),
            tf.keras.layers.Dense(50, activation='elu'),
            tf.keras.layers.Dense(self.point_predis)
        ])

    def configure(self, optimiseur, loss, unite):
        # Configuration du modèle(on minimise avec la méthode des moindres carrés)
        self.modele_lstm.compile(optimizer='adam', metrics=['mae'], loss='mse')

    def entrainement(self, modele, EPOQUE, X_entrainement, Y_entrainement):
        # Lance l'entrainement du modèle
        history = modele.fit(X_entrainement, Y_entrainement, epochs=EPOQUE)
        return history

    def prediction(self, X_test, Y_test):
        # Prédis le paramètre sur la station test
        YPred = self.modele_lstm.predict(X_test, verbose=0)
        Y_test = Y_test.reshape(Y_test.shape[0], self.point_predis, )
        # YPred = denormaliser(YPred,meteo_test_non_norm,'td')
        # Y_test = denormaliser(Y_test,meteo_test_non_norm,'td')
        # X_test = denormaliser(X_test,meteo_test_non_norm,'td')
        return Y_test, YPred

    # V.Affichage des résultats

    def formatage_pred(self, YPred, Y_test):
        # Listes contenant les valeurs des prédictions
        Liste_finale = []
        Valeurs_liste = []

        # Sur les 50 valeurs prédites on en prend une(ici 40 donc 40*)pour chaque segment sur les 3 ans
        for i in YPred:
            Liste_finale.append(i[40])

        np_array = np.array(Liste_finale)

        for i in Y_test:
            Valeurs_liste.append(i[40])

        val_np_array = np.array(Valeurs_liste)

        return val_np_array, np_array

    def simple_plot(self, np_array, val_np_array):
        # Première figure: La prédiction comparé au réel sur les 3 ans avec
        plt.figure(figsize=(30, 5))
        sns.set(rc={"lines.linewidth": 3})
        sns.lineplot(x=np.arange(val_np_array.shape[0]), y=val_np_array, color="green")
        sns.set(rc={"lines.linewidth": 3})
        sns.lineplot(x=np.arange(np_array.shape[0]), y=np_array, color="coral")
        plt.margins(x=0, y=0.5)
        plt.legend(["Original", "Prédiction"])

    def multi_plot(self, X_test, Y_test, YPred):
        # Deuxième figure: Une prediction particulière avec les 100 données et les 50 prédictions.
        self.plot_multi_etape(X_test[136], Y_test[136], YPred[136])
        # Aucune échelle est bonne il faut faire l'inverse de la noramalisation si on veut une bonne échelle.


