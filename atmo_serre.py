import numpy as np
from math import *


sigma = 5.6704*10**(-8) #constante de Boltzmann

def coef_moy():  #Pour T = +15°C
    X = (15+273)**4*sigma
    return (X-240)/X     #X = puissance émise par la terre

# print(coef_moy())

C_CO2_moy = 400 #ppm
C_H2O_moy = 25000 #ppm
coef_moy = coef_moy()


def super_calcul(C_CO2, C_H2O = C_H2O_moy):
    coef = 0.25*coef_moy + 0.25*coef_moy*(C_CO2/C_CO2_moy)**(1/2.6) + 0.5*coef_moy*(C_H2O/C_H2O_moy)**(1/2.6)
    X = 240/(1-coef)
    print((X/sigma)**(1/4)-273-13.9)
    return coef


super_calcul(316, C_H2O_moy)
super_calcul(325, C_H2O_moy)
super_calcul(338, C_H2O_moy)
super_calcul(354, C_H2O_moy)
super_calcul(369, C_H2O_moy)
super_calcul(389, C_H2O_moy)
super_calcul(415, C_H2O_moy)



"""
Explication du modèle :

Nous cherchons ici à moduler la température en fonction de la concentration de CO2 dans l'air. Nous nous basons sur l'effet de serre.

Nous avons modélisé l'effet de serre ainsi : il y a une certaine puissance émise par la terre (ici notée X). Cette puissance dépend de la température du sol de la terre, donc si nous avons cette puissance, nous avons la température (avec la formule de Boltzmann). Nous considérons qu'une partie de cette puissance va être absorbée par les gaz à effet de serre, et une partie va sortir dans l'espace. La puissance sortant de l'atmosphère est égale à celle entrant, soit 240W/m2. Il nous reste plus qu'à avoir la proportion puissance absorbée/puissance émise, notée coef, pour avoir la puissance émise.

Nous cherchons donc ici à trouver coef.

Nous avons commencé par un calcul très simple d'un coef moyen, coef_moy, qui correspond à une température au sol de 15°C.
Nous calculons ensuite le coefficient en modifiant ce coef_moy. Nous avons écris l'équation : coef = 0.25*coef_moy + 0.25*coef_moy*(C_CO2/C_CO2_moy) + 0.5*coef_moy*(C_H2O/C_H2O_moy). Ici, le 0.25 et 0.5 correspondent à un coeficient qui pondère l'effet des différents GES sur l'effet de serre. Nous avons multiplié par C_CO2/C_CO2_moy et C_H2O/C_H2O_moy pour que aux concentration moyenne, on est le coefficient moyen. On a bien la température qui agmente lorsque la concentration en GES augmente.
Cependannt, la formule ne se basant sur aucune preuve scientifique, nous avons modifié petit à petit le coefficient à la puissance de C_CO2/C_CO2_moy pour nous rapprocher au mieux du tableau de données si dessous. Nous avons donc validé ce modèle non pas avec des preuves scientifiques, mais parce qu'il marche.

 Augmentation de la température (°C) | Concentration de CO2 (ppm) |
 ------------------------------------|-----------------------------|
 0                                   | 316                         |
 0.1                                 | 325                         |
 0.3                                 | 338                         |
 0.6                                 | 354                         |
 0.8                                 | 369                         |
 1.0                                 | 389                         |
 1.2                                 | 415                         |

"""
