#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import datetime
import os.path
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
    
nltk.download('punkt')

ruta = os.path.abspath(os.path.dirname(__file__))
datosEntrenamiento = os.path.join(ruta,"datosEntrenamiento.txt")
datosEntrada = os.path.join(ruta,"datosEntrada.txt")
synapse_file = os.path.join(ruta,"synapses.json")

print ("\n"
"\n ---------------------------------------------"
"\n|  INSTITUTO TECNOLÓGICO SUPERIOR DE HUETAMO  |"
"\n ---------------------------------------------"
"\n"
"\n ========================================"
"\n|    Dr. Carlos Reyes Dueñas             |"
"\n|    Dra. Mariela Yanin Magaña Gutiérrez |"
"\n ========================================"
"\n")

# Clases de datos
# Se cargan los datos de entrenamiento a partir del archivo
# Presentacion o inicio de conversacion
training_data = []
with open(datosEntrenamiento) as f:
    for line in f:
       (clase, texto) = line.split("\t>\t")
       training_data.append({"class":clase,"sentence":texto.replace("\n", "")})

# print ("Datos de entrenamiento:", training_data)
print ("%s sentencias agregadas a datos de entrenamiento" % len(training_data))


words = []
classes = []
documents = []
ignore_words = ['?',',',';','.',':','...']
# Ciclo para recabar datos de entrenamiento
for pattern in training_data:
    # tokenizamos cada palabra de las sentencias
    w = nltk.word_tokenize(pattern['sentence'])
    # agregamos a la lista de palabras
    words.extend(w)
    # agregamos a nuestros documentos la lista de palabras que pertenecen a determinada categoria
    documents.append((w, pattern['class']))
    # agregamos a nuestra lista de categorias
    if pattern['class'] not in classes:
        classes.append(pattern['class'])

# convertimos en minusculas cada palabra y removemos duplicados
words = [w.lower() for w in words if w not in ignore_words]
words = list(set(words))

# removemos clases duplicadas
classes = list(set(classes))

print ("%s Categorias de palabras detectadas" % len(classes))
print ("%s Palabras unicas identificadas en el catalogo" % len(words))

#==========================================

# creamos una lista para datos de entrenamiento
training = []
#Creamos lista para guardar nuestras salidas
output = []

output_empty = [0] * len(classes)

# se asignaran los datos de entrenamiento, y al mismo tiempo
# asignamos 1 al grupo de palabras validas para cada sentencia, 0 cuando no es asi
for doc in documents:
    # inicializamos el grupo de palabras validas, que resultara por ejemplo: [0,0,0,1,0] 
    bag = []
    # lista palabras obtenidas por patron
    pattern_words = doc[0]
    # convertimos todo en minusculas
    pattern_words = [word.lower() for word in pattern_words]
    # Creamos la numeracion de palabras validas [0,0,0,1,0] por ejemplo
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    training.append(bag)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    output.append(output_row)


# -----------------------------------------------
# 

import numpy as np
import time

#  sigmoid 
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

    # sigmoid derivada
def sigmoid_output_to_derivative(output):
    return output*(1-output)
 
def clean_up_sentence(sentence):
    # tokeniza el padron de setencias
    sentence_words = nltk.word_tokenize(sentence)
    # convercion a minusculas
    sentence_words = [word.lower() for word in sentence_words]
    return sentence_words

# retornamos el valor de las palabras reconocidas: 0 o 1 para cada palabra en el grupo de palabras recabadas anteriormente
def bow(sentence, words, show_details=False):
    # tokeniza las sentencias
    sentence_words = clean_up_sentence(sentence)
    # reconocimiento de palabras reconocidas
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("Se encontro como activa la palabra: %s" % w)
    return(np.array(bag))

def think(sentence, show_details=False):
    x = bow(sentence.lower(), words, show_details)
    if show_details:
        print ("sentencia:", sentence, "\n proyeccion:", x)
    # datos de entrada del grupo de palabras reconocidas 0 o 1
    l0 = x    
    # Multiplicacion de matriz de entrada y capas ocultas
    l1 = sigmoid(np.dot(l0, synapse_0))
    # salida
    l2 = sigmoid(np.dot(l1, synapse_1))
    return l2

def train(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):

    print ("Trabajando entrenamiento con %s neuronas" % hidden_neurons )
    print ("Dimensiones de matriz de entrada: %sx%s    Numero de categorias a reconocer: %s" % (len(X),len(X[0]), len(classes)) )
    np.random.seed(1)

    last_mean_error = 1
    # inicializacion de pesos
    synapse_0 = 2*np.random.random((len(X[0]), hidden_neurons)) - 1
    synapse_1 = 2*np.random.random((hidden_neurons, len(classes))) - 1

    prev_synapse_0_weight_update = np.zeros_like(synapse_0)
    prev_synapse_1_weight_update = np.zeros_like(synapse_1)

    synapse_0_direction_count = np.zeros_like(synapse_0)
    synapse_1_direction_count = np.zeros_like(synapse_1)
        
    for j in iter(range(epochs+1)):

        # asignacion de valores a las capas 0, 1 y 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))
                
        if(dropout):
            layer_1 *= np.random.binomial([np.ones((len(X),hidden_neurons))],1-dropout_percent)[0] * (1.0/(1-dropout_percent))

        layer_2 = sigmoid(np.dot(layer_1, synapse_1))

        # ajuste de error para con el objetivo y
        layer_2_error = y - layer_2

        #Grupo de 10000 iteraciones
        if (j% 10000) == 0 and j > 5000:    
            # Si el error actual es mayor que el ultimo error, entonces interrumpe
            if np.mean(np.abs(layer_2_error)) < last_mean_error:
                print ("Entrenando... "+str(j)+" iteraciones: " + str(np.mean(np.abs(layer_2_error))) )
                last_mean_error = np.mean(np.abs(layer_2_error))
            else:
                print ("break:", np.mean(np.abs(layer_2_error)), ">", last_mean_error )
                break
                        
        # ajuste de error        
        layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

        # pero de error
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # ajuste del error
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
        
        synapse_1_weight_update = (layer_1.T.dot(layer_2_delta))
        synapse_0_weight_update = (layer_0.T.dot(layer_1_delta))
        
        if(j > 0):
            synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0)+0) - ((prev_synapse_0_weight_update > 0) + 0))
            synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0)+0) - ((prev_synapse_1_weight_update > 0) + 0))        
        
        synapse_1 += alpha * synapse_1_weight_update
        synapse_0 += alpha * synapse_0_weight_update
        
        prev_synapse_0_weight_update = synapse_0_weight_update
        prev_synapse_1_weight_update = synapse_1_weight_update

    now = datetime.datetime.now()

    # preparacion de almacenamiento de sinapsis
    synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
               'datetime': now.strftime("%Y-%m-%d %H:%M"),
               'words': words,
               'classes': classes
              }
    

    with open(synapse_file, 'w') as outfile:
        json.dump(synapse, outfile, indent=4, sort_keys=True)
    print ("se guardaron las sinapsis en: ", synapse_file)

#==========================================

# valor minimo para tolerancia de error
TOLERANCIA = 0.3
ITERACIONES_ENTRENAMIENTO = 100000

X = np.array(training)
y = np.array(output)

start_time = time.time()
print (""
"----------------------------------------------------------------------------"
 "ENTRENANDO..."
 "----------------------------------------------------------------------------"
 "")

train(X, y, hidden_neurons=20, alpha=0.1, epochs=ITERACIONES_ENTRENAMIENTO, dropout=False, dropout_percent=0.2)

elapsed_time = time.time() - start_time
print ("Tiempo de procesamiento: ", elapsed_time, " segundos")

print (""
"----------------------------------------------------------------------------"
 "ENTRENAMIENTO FINALIZADO...CARGANDO DATOS DE ENTRADA"
 "----------------------------------------------------------------------------"
 "")
# se cargan las sinapsis calculadas

with open(synapse_file) as data_file: 
    synapse = json.load(data_file) 
    synapse_0 = np.asarray(synapse['synapse0']) 
    synapse_1 = np.asarray(synapse['synapse1'])

def classify(sentence, show_details=False):
    results = think(sentence, show_details)

    results = [[i,r] for i,r in enumerate(results) if r>TOLERANCIA ] 
    results.sort(key=lambda x: x[1], reverse=True) 
    return_results =[[classes[r[0]],r[1]] for r in results]
    print ("%s -> Categoria: %s \n" % (sentence, return_results))
    return return_results

# Cargamos los datos de entrada que pondran a prueba
with open(datosEntrada) as f:
    for line in f:       
        classify(line)
