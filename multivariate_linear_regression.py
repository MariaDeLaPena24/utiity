"""
Este script de Python implementa el descenso del gradiente para estimar los parametros de w, 
los cuales se utilizan para generar la regresion lineal multivariable y poder obtener un modelo 
representativo de los datos. 
Se utiliza la librería de numpy para generar los arreglos y las operaciones que se les aplican.
Tambien se utiliza la libreria de utilityfuntions que fue creada por nosotros mismos y ayuda en 
el manejo de los datos. 

Autores: Maria Fernanda de la Peña Cuevas e Isaias Gomez Lopez
Emails: maria.delapena@udem.edu
        isaias.gomez@udem.edu
Institution: Universidad de Monterrey
First created: 5 de noviembre del 2020
"""
#Importar librerias estandar
import time
import numpy as np

#Importar libreria creada
import utilityfunctions as uf

#Funcion principal
def main():
    """
    Descripcion:
        Funcion principal del programa, por lo que aqui se encuentra el codigo principal.
        Se iran llamando a otras funciones y librerias conforme se necesite y se imprimiran los resultados.
        Realiza la funcion del gradiente descendiente para regresion lineal multivariable. 
    Inputs:
        None
    Outputs:
        None
    """
    #Variable para determinar si se imprime los datos de prueba
    flag = 1
    
    #Extrae la informacion de un archivo determinado, consigue los valores de X, Y, la media y la distribuccion estandar
    X_train, y_train, mean, std = uf.load_data_multivariate('training-data-multivariate.csv', flag)
    
    #Normaliza los valores de x
    X_train = uf.normalization(X_train, 'Training data scaled', mean, std)

    #Inicializa el parametro w
    w = np.zeros((X_train.shape[1]+1,1), dtype = float)

    #Se inicializa el learning_rate y el valor de paro
    stopping_criteria = 0.01
    learning_rate = 0.5

    #Se aplica la funcion del descenso gradiente
    w, iterations = uf.gradient_descent_multivariate(X_train, y_train,w,stopping_criteria,learning_rate)

    #Se imprimen los parametros w
    uf.print_parameters(w)

    #Se inicializan los valores de prueba
    X_testing = np.array([[24.51, 0.34, 50.2, 2.78], 
                        [33.98, 0.62, 50.1, 9.79], 
                        [9.57, 0.32, 52.4, 3.15]])


    #Variable para determinar si se imprime los datos de prueba
    flag = 1

    #Prediccion de los valores de acuerdo al modelo obtenido. 
    #Se normalizan los datos
    price = uf.predict(w, X_testing, mean, std, flag, 'Testing data scaled')

    #Impresion de los resultados obtenidos de la prediccion
    uf.print_predictions(price, 'Last mile cost [predicted]:')

    #Impresion de los valores
    print('Learning rate:', learning_rate)
    print('Running time:', time.process_time())
    print('Number of iterations:', iterations)

#Se llama a la funcion principal
main()