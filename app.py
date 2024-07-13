#perceptron: hidden layer'a sahip olmayan, tek bir main neuron'a sahip(işlem kısmı olarak) yapay sinir ağlarına verilen isim.

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1-x)

training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

print("Girdi değerleri: ", training_inputs )

"""
transpoz işlemi yapılarak
[0,1,1,0] ifadesi

[
  0,
  1,
  1,
  0
]
'a dönüştürüldü
"""
training_outputs = np.array([[0,1,1,0]]).T

#rastgele sayı üretme algoritmasına başlangıç değeri olarak 1 değerini göndererek, algoritma her çalıştığında başlangıç değerini algoritmada kullanacağı için aynı rastgele sayıların üretilmesini sağlar
np.random.seed(1)

#rastgele sayıların oluşması için bir denklem oluşturduk
synaptic_weights = 2 * np.random.random((3,1)) - 1
print("Rastgele synaptic başlangıç değerleri: ", synaptic_weights)


for iteration in range(200000):
    #girdi olarak eğitim setini kullanacağımızı belirttik
    input_layer = training_inputs
    #input_layer matrisi ile synaptic_weight matrisini çarptık ve sonucu sigmoid fonksiyonundan geçirerek değerlerin 0 ile 1 arasında olmasını sağladık.
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    
    #YAPAY SİNİR AĞININ EĞİTİLMESİ
    #"error weight derivative" yöntemi ile backpropagation işlemi
    error = training_outputs - outputs
    adjustments = error * sigmoid_derivative(outputs)

    synaptic_weights += np.dot(input_layer.T, adjustments)

print("Eğtiim sonrası Synaptic weights değerleri: ", synaptic_weights)

print("Çıktı değerleri: ", outputs)


