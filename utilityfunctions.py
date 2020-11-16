"""
Este script de Python es una libreria que contiene las funciones que se necesitan para 
realizar el gradiente descendiente con regresion lineal multivariable.
Se necesitan las librerias de pandas y de numpy.

Autores: Maria Fernanda de la Peña Cuevas e Isaias Gomez Lopez
Emails: maria.delapena@udem.edu
        isaias.gomez@udem.edu
Institution: Universidad de Monterrey
First created: 5 de noviembre del 2020
"""
#Importacion de librerias estandar
import pandas as pd
import numpy as np

#Funciones que contiene esta libreria
def eval_hypothesis_function_multivariate(w, x):
    """
    Descripcion:
    Esta funcion de utiliza para evaluar la funcion hipotesis
    Inputs:
        x: arreglo estilo numpy con los valores en x
        w: arreglo estilo numpy con los valores de los parametros w
    Outputs:
        return: arreglo estilo numpy con el valor de la funcion hipotesis
    """
    #Regresa el valor de la funcion de hipotesis
    return np.dot(x, w)

def compute_gradient_of_cost_function_multivariate(x, y, w):
    """
    Descripcion:
    Esta funcion de utiliza para calcular el valor del gradiente de la funcion
    de costo
    Inputs:
        x: arreglo estilo numpy con los valores en x
        y: arreglo estilo numpy con los valores en y
        w: arreglo estilo numpy con los valores de los parametros w
    Outputs:
        gradient: arreglo estilo numpy con el valor del gradiente
    """
    #Calcula la forma de x
    N = np.shape(x)
    #Cambiar la forma de y para que sea una matriz
    y = np.reshape(y,[y.shape[0],1])
    #Calcula la diferencia entre lo predicho y lo real
    Ydef = eval_hypothesis_function_multivariate(w, x) - y
    #Calcula el valor del gradiente
    gradient = np.sum(np.multiply(Ydef, x), axis = 0)/N[0]
    #Cambia la forma para que sea una matriz
    gradient=np.reshape(gradient,[gradient.shape[0],1])
    #Regresa el valor del gradiente
    return gradient

def compute_L2_norm_multivariate(gradient_of_cost_function):
    """
    Descripcion:
    Esta funcion de utiliza para calcular el valor del gradiente de la norma L2.
    Tambien conocido como la norma euclideana. 
    Inputs:
        gradient_of_cost_function: arreglo estilo numpy con el valor del gradiente
                                   de la funcion de costo
    Outputs:
        L: arreglo estilo numpy con el valor de la norma L2
    """
    #Calcula el valor de la norma L2
    L = np.sqrt(np.matmul(gradient_of_cost_function.T, gradient_of_cost_function))
    #Retorna el valor de la norma L2 obtenido
    return L

def load_data_multivariate(path_and_filename, flag):
    """
    Descripcion:
    Esta funcion de utiliza para obtener los datos de entrenamiento a partir de un archivo.
    Inputs:
        path_and_filename: variable tipo string con el nombre del archivo a importar
        flag: variable tipo entero que sirve para indicar si se imprime los datos de 
              entrenamiento
    Outputs:
        X: arreglo estilo numpy con los valores de x obtenidos del archivo leido
        y: arreglo estilo numpy con los valores de y obtenidos del archivo leido
        X_mean: arreglo estilo numpy con los valores de las medias por columna de x
        X_std: arreglo estilo numpy con los valores de las desviaciones estandar por columna de x
    """
    #Funcion para comprobar que existe el archivo y se pueda realizar lo demas
    try:
        #Importacion del archivo
        Training_data = pd.read_csv(path_and_filename)
        #Obtener el nombre de las columnas
        columnas = Training_data.columns
        #Comparador para saber si es necesario imprimir los datos
        if flag == 1:
            print(70*'-', '\n Training data and Y outputs')
            print(70*'-')
            print(Training_data)
            print(70*'-')
        #Separar los datos en los arreglos X y Y
        X = np.array(Training_data[columnas[:-1]])
        y = np.array(Training_data[columnas[-1]])
        #Calcular la media y desviacion estandar de x.
        X_mean, X_std = X.mean(axis = 0), X.std(axis = 0)
        
        #Retorna el valor de X, y, la media de las X y la desviacion estandar de las X
        return X, y, X_mean, X_std
    #Si no se puede encontrar el archivo, se genera un error y se imprime un enunciado
    except IOError:
        print('Archivo no encontrado. Se recomienda revisar el archivo ingresado')


def gradient_descent_multivariate(x_training, y_training, w, stopping_criteria, learning_rate):
    """
    Descripcion:
    Esta funcion de utiliza para calcular la regresion lineal multivariable con gradiente descendiente
    Inputs:
        x_training: arreglo estilo numpy con los valores en x
        y_training: arreglo estilo numpy con los valores en y
        w: arreglo estilo numpy con los valores de los parametros w
        stopping_criteria: variable tipo flotante que servira para parar el algoritmo. 
                           Se para cuando sea menor a este valor
        learning_rate: variable tipo flotante que representa el tamano de los saltos 
    Outputs:
        w: arreglo estilo numpy con los valores actualizados de w.
        count: variable tipo entero que guarda el numero de iteraciones que realiza el algoritmo. 
    """
    #Valor inicial para la norma L2
    L2_norm = 100
    #Agregar una columna de 1 al arreglo de las x
    x_training = np.c_[np.ones(x_training.shape[0]), x_training]
    #Iniciar el contador de iteraciones
    count = 0
    #Comienza el lazo para calcular los parametros de w
    while L2_norm > stopping_criteria:
        #Calcular el valor del gradiente
        gradient = compute_gradient_of_cost_function_multivariate(x_training, y_training, w)
        #Actualizar el valor de w
        w = w - learning_rate*gradient
        #Actualizar el valor de la norma L2
        L2_norm = compute_L2_norm_multivariate(gradient)
        #Actualizar el contador
        count += 1
    #Retorna w y el contador
    return w, count


def predict(w,x,mean,std,flag,phrase):
    """
    Descripcion:
    Esta funcion de utiliza para probar el modelo obtenido con datos nuevos
    Inputs:
        w: arreglo estilo numpy con los valores de los parametros w
        x: arreglo estilo numpy con los valores en x
        mean: arreglo estilo numpy con los valores de las medias de x
        std: arreglo estilo numpy con los valores de las desviaciones estandar de x
        flag: variable tipo entero que sirve para indicar si se imprime los valores de prueba
        phrase: variable tipo string que se usara para imprimir con los datos de prueba

    Outputs:
        return: arreglo estilo numpy con los valores predecidos
    """
    #Verifica si se deben imprimir los datos
    if flag == 1:
        print(70*'-', '\n Testing data')
        print(70*'-')
        print(x)
        print(70*'-')
    #normaliza los datos de prueba
    x_norm = normalization(x, phrase, mean, std)
    #Agrega una columna de 1 a los datos normalizados
    x_norm = np.c_[np.ones(x.shape[0]),x_norm]
    
    #Regresa los valores predecidos
    return np.dot(x_norm,w)

def normalization(X, phrase, X_mean, X_std):
    """
    Descripcion:
    Esta funcion de utiliza para normalizar los datos
    de costo
    Inputs:
        X: arreglo estilo numpy con los valores a ser normalizados
        phrase: variable tipo string que se imprimira con los valores normalizados
        X_mean: arreglo estilo numpy con los valores de la media de x
        X_std: arreglo estilo numpy con los valores de la desviacion estandar de x
    Outputs:
        Xnorm: arreglo estilo numpy con los valores normalizados
    """
    #Normaliza los datos
    Xnorm = (X - X_mean)/np.sqrt(X_std**2 + 10**-8)

    #Imprime los datos normalizados con la frase personalizada
    print('{}'.format(phrase))
    print(70*'-')
    print(Xnorm)
    print(70*'-')

    #Retorna los datos normalizados
    return Xnorm

def print_parameters(W):
    """
    Descripcion:
    Esta funcion se utiliza para imprimir los valores de los parametros w
    Inputs:
        w: arreglo estilo numpy con los valores de los parametros w
    Outputs:
        None
    """
    #Imprime los parametros
    print('W parameters: ') 
    print(70*'-')
    for i, w, in enumerate(W):
        print('w{} = {}'.format(i, w))
    print(70*'-')

def print_predictions(price, phrase):
    """
    Descripcion:
    Esta funcion se utiliza para imprimir los valores de las predicciones
    Inputs:
        price: arreglo estilo numpy con los valores de las predicciones
        phrase: variable tipo string que se usara para imprimir con los datos de prueba
    Outputs:
        None
    """
    #Imprime los datos predecidos con su frase personalizada
    print(70*'-', '\n{}'.format(phrase))
    print(70*'-')
    print(price)
    print(70*'-') 
