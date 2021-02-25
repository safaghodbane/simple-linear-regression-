import numpy as np                #pour calcul mathématique
import matplotlib.pyplot as plt   #visualisation des données
import pandas as pd               #traitement des données 



dataset = pd.read_csv('Salary_Data.csv')  #Importer et lire  dataset
x = dataset.iloc[:,:-1].values     #1er colone de data
y = dataset.iloc[:,-1].values     #2 eme colone de data


from sklearn.model_selection import train_test_split  #pour fractionner data en partie appentissage et partie test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0) #(1/3) des valeur pour test #pour prendre des valeurs aléatoires 


from sklearn.linear_model import LinearRegression   #importer le model linear regression
regressor = LinearRegression()

regressor.fit(x_train, y_train) #trouver le meilleur model pour le trassage de la ligne 
                                #trainer la machine

y_pred = regressor.predict(x_test)  #Prédire y de la resultat de test de x 
                                    #on compare y-pred et y-test
  
        
  

plt.scatter(x_train, y_train, color = 'red')  #visualisation des données de "training set" (trassage de courbe "taining set" )
plt.plot(x_train, regressor.predict(x_train), color = 'blue')#trassage de courbe "taining set" 
plt.title('Salary vs Experience (Training set)') #donner un titre de figure
plt.xlabel('Years of Experience') #donner un titre au axe des abscisses
plt.ylabel('Salary') #donner un titre au axe des ordonnées
plt.show() #





plt.scatter(x_test, y_test, color = 'red')  #dessiner les points en couleur rouge 
plt.plot(x_train, regressor.predict(x_train), color = 'blue') #trassage de courbe "test set" 
plt.title('Salary vs Experience (Test set)') #donner un titre au figure
plt.xlabel('Years of Experience') # donner un titre au axe des abscisses
plt.ylabel('Salary') #donner un titre au axe des ordonnées
plt.show()  #





y_pred2= regressor.predict([[15]]) #essayer de prédire le salaire d'un employer qui a 15 ans d'experience
print(y_pred2)
print("----------------------") 
print('Train Score: ', regressor.score(x_train, y_train)) #le score de training
print('Test Score: ', regressor.score(x_test, y_test)) #le score de test 
