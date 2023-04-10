# I.importation


# algèbre linéaire
import numpy as np

# transforamtion des données et importation depuis le csv
import pandas as pd

# repésentation des résultats
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
import tensorflow as tf

# II.fonction utile

# 14578001(Deauville)
station_meteo = 14578001
# 22005003(Belle-Isle-en-Terre(Côtes-d'armor))
test_station_meteo = 22005003


def normaliser(dataset, parametre, parametre_unique=False):
    if parametre_unique:
        dataNorm = dataset
        dataNorm[parametre] = ((dataset[parametre] - dataset[parametre].min()) / (
                    dataset[parametre].max() - dataset[parametre].min()))
        return dataNorm
    else:
        dataNorm = ((dataset - dataset.min()) / (dataset.max() - dataset.min()))
        #         dataNorm[target]=dataset[target]
        return dataNorm


def denormaliser(dataset, dataset_origine, parametre, parametre_unique=False):
    if parametre_unique:
        dataNorm = dataset
        dataNorm[parametre] = (
                    (dataset[parametre]) * (dataset_origine[parametre].max() - dataset_origine[parametre].min()) + (
                dataset_origine[parametre].min()))
        return dataNorm
    else:
        dataNorm = ((dataset) * (dataset_origine.max() - dataset_origine.min()) + (dataset_origine.min()))
        #       dataNorm[target]=dataset[target]
        return dataNorm


def segment(dataset, variable, intervale=5000, futur=0):
    data = []
    labels = []
    for i in range(len(dataset)):
        debut_index = i
        fin_index = i + intervale
        futur_index = i + intervale + futur
        if futur_index >= len(dataset):
            break
        data.append(dataset[variable][i:fin_index])
        labels.append(dataset[variable][fin_index:futur_index])
    return np.array(data), np.array(labels)


def temp_int(longeur):
    return list(range(-longeur, 0))


def plot_multi_etape(historique, vrai_futur, prediction):
    plt.figure(figsize=(12, 6))
    num_passe = temp_int(len(historique))
    num_futur = len(vrai_futur)
    plt.plot(num_passe, np.array(historique[:, 0]), label='Passé donné')
    plt.plot(np.arange(num_futur), np.array(vrai_futur), label='Véritable futur')
    if prediction.any():
        plt.plot(np.arange(num_futur), np.array(prediction), 'ro', label='Futur prédis')
    plt.legend(loc='upper left')
    plt.show()


# III.Transformation des données


# importation de notre dataset sous panda
df2016 = pd.read_csv(r'/kaggle/input/meteonet/NW_Ground_Stations/NW_Ground_Stations/NW_Ground_Stations_2016.csv')
df2017 = pd.read_csv(r'/kaggle/input/meteonet/NW_Ground_Stations/NW_Ground_Stations/NW_Ground_Stations_2017.csv')
df2018 = pd.read_csv(r'/kaggle/input/meteonet/NW_Ground_Stations/NW_Ground_Stations/NW_Ground_Stations_2018.csv')


def index(station):
    data = df2016[(df2016['number_sta'] == station)]
    data = data.append(df2017[(df2017['number_sta'] == station)], ignore_index=True)
    data = data.append(df2018[(df2018['number_sta'] == station)], ignore_index=True)
    data['date'] = pd.to_datetime(data['date'], format='%Y%m%d %H:%M')
    data.set_index('date', inplace=True)
    data['td'] = data['td'].interpolate('linear')
    data['precip'] = data['precip'].interpolate('linear')
    data['hu'] = data['hu'].interpolate('linear')
    data['ff'] = data['ff'].interpolate('linear')
    data = data.drop(['number_sta', 'lat', 'lon', 'height_sta'], axis=1)
    return data


# création de nos données
meteo_non_norm = index(station_meteo)
meteo_test_non_norm = index(test_station_meteo)

# normalisation pour accélerer le calcul
meteo = normaliser(meteo_non_norm, 'td', parametre_unique=False)
meteo_test = normaliser(meteo_test_non_norm, 'td', parametre_unique=False)

# on prend la moyenne sur des périodes de 720 messures(5 min) = 12 heures
meteo_cp = meteo.resample('720T').mean()
meteo_test_cp = meteo_test.resample('720T').mean()

# on considère la température constante si on a pas de données :
meteo_cp = meteo_cp.fillna(method='bfill')
meteo_test_cp = meteo_test_cp.fillna(method='bfill')

# IV.Création du modèle


# Selection du nombre de point donné et ceux à prédir
point_hist = 100
point_predis = 50


# Prépare des sacs de données suivant la sélection et imprime la Dimension des données (permet de voir si le volume de données est trop faible ou trop grand)
def segmentation(dataset, parametre):
    X, Y = segment(dataset, parametre, intervale=point_hist, futur=point_predis)
    X = X.reshape(X.shape[0], point_hist, 1)
    Y = Y.reshape(Y.shape[0], point_predis, 1)
    print("Dimension du Passé: ", X.shape)
    print("Dimension du Futur: ", Y.shape)
    return X, Y


X_entrainement, Y_entrainement = segmentation(meteo_cp, 'td')
X_test, Y_test = segmentation(meteo_test_cp, 'td')

# Nombre d'époque d'entrainement(résultat satisfaisant vers 200)
EPOQUE = 200

# modèle du réseaux de neurones(4 rangées (100,100,50,50) dont la première LSTM)
modele_lstm = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(point_hist, input_shape=X_entrainement.shape[-2:]),
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(50),
    tf.keras.layers.Dense(point_predis)
])

# Configuration du modèle(on minimise avec la méthode des moindres carrés)
modele_lstm.compile(optimizer='adam', metrics=['mae'], loss='mse')

# Lance l'entrainement du modèle
modele_lstm.fit(X_entrainement, Y_entrainement, epochs=EPOQUE)

# Prédis le paramètre sur la station test
YPred = modele_lstm.predict(X_test, verbose=0)
Y_test = Y_test.reshape(Y_test.shape[0], point_predis, )

# V.Affichage des résultats


# Listes contenant les valeurs des prédictions
Liste_finale = []
Valeurs_liste = []

# Sur les 50 valeurs prédites on en prend une(ici 40 donc 40*)pour chaque segment sur les 3 ans
for i in YPred:
    Liste_finale.append(i[40])

np_array = np.array(Liste_finale)
print(np_array.shape)

for i in Y_test:
    Valeurs_liste.append(i[40])

val_np_array = np.array(Valeurs_liste)
val_pd_array = pd.DataFrame(val_np_array)
# denormaliser(val_pd_array,meteo_non_norm,'td',parametre_unique=False)

print(val_np_array.shape)

# Première figure: La prédiction comparé au réel sur les 3 ans avec
plt.figure(figsize=(30, 5))
sns.set(rc={"lines.linewidth": 3})
sns.lineplot(x=np.arange(val_np_array.shape[0]), y=val_np_array, color="green")
sns.set(rc={"lines.linewidth": 3})
sns.lineplot(x=np.arange(np_array.shape[0]), y=np_array, color="coral")
plt.margins(x=0, y=0.5)
plt.legend(["Original", "Prédiction"])

# Deuxième figure: Une prediction particulière avec les 100 données et les 50 prédictions.
plot_multi_etape(X_test[1336], Y_test[1336], YPred[1336])
# Aucune échelle est bonne il faut faire l'inverse de la noramalisation si on veut une bonne échelle.
