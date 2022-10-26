
import copy
import random
from tkinter import E
import numpy as np
from sklearn import svm
import cv2 as cv
import matplotlib.pyplot as plt
from sklearn import metrics
from perceptron import Perceptron

# No se esta usando


def generate_set(n):
    pairs = np.dstack(np.meshgrid(np.arange(5), np.arange(5))).reshape(-1, 2)
    # now select a random set of 25 of those pairs, which are unique
    return pairs[np.random.choice(np.arange(pairs.shape[0]), size=n, replace=False)]

# Función lineal.


def f2(x):
    return x + 1


n = 20
TP3_1 = [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 4],
         [1, 4, 4],
         [1, 4, 3],
         [1, 0, 4],
         [1, 4, 2],
         [1, 4, 1],
         [1, 1, 1],
         [1, 0, 1],
         [1, 3, 3],
         [1, 3, 2],
         [1, 0, 3],
         [1, 2, 4],
         [1, 3, 4],
         [1, 4, 0],
         [1, 2, 0],
         [1, 2, 3],
         [1, 1, 3],
         [1, 1, 2]]





def show_confusion_matrix(title  ):
    confusion_matrix = metrics.confusion_matrix()

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    cm_display.plot()
    plt.title(title)
    plt.ylabel('Clase resultado')
    plt.xlabel('Clase verdadera')
    plt.show()
def get_results(arr, fn):
    return list(map(lambda x: -1 if (x[1] > fn(x[0])) else 1, arr))


def generate_img(list_x, list_y, y, fn_perceptron, W, name, title):
    xo, yo, xg, yg = get_xy(y, list_x, list_y)
    plt.plot(xo, yo, "o", color="red")
    # Plot de todos los Xi con Y = 1
    plt.plot(xg, yg, "o", color="orange")

    # Graficar ambas funcion.
    plt.plot(list_x, [fn_perceptron(W, x) for x in list_x])
    # Establecer el color de los ejes.
    plt.axhline(0, color="black")
    plt.axvline(0, color="black")
    # Limitar los valores de los ejes.
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    # Guardar gráfico como imágen PNG.
    plt.title(title)
    plt.savefig(name + ".png")
    plt.show()


TP3_withouth_x0 = list(map(lambda x: [x[1], x[2]], TP3_1))
y = np.array(get_results(TP3_withouth_x0, f2))


# Graficar puntos
list_x = list(map(lambda x: x[0], TP3_withouth_x0))
list_y = list(map(lambda x: x[1], TP3_withouth_x0))


def get_xy(y, list_x, list_y):
    xc1 = []
    yc1 = []
    xc2 = []
    yc2 = []
    for i in range(len(y)):
        if (y[i] == -1):
            xc1.append(list_x[i])
            yc1.append(list_y[i])
        else:
            xc2.append(list_x[i])
            yc2.append(list_y[i])
    return xc1, yc1, xc2, yc2


xo, yo, xg, yg = get_xy(y, list_x, list_y)
plt.plot(xo, yo, "o", color="red")
# Plot de todos los Xi con Y = 1
plt.plot(xg, yg, "o", color="orange")




# Graficar funcion.
plt.plot(list_x, [f2(i) for i in list_x], "blue")


# Establecer el color de los ejes.
plt.axhline(0, color="black")
plt.axvline(0, color="black")
# Limitar los valores de los ejes.
plt.xlim(0, 5)
plt.ylim(0, 5)
# Guardar gráfico como imágen PNG.
plt.title("Recta creada a mano")
plt.savefig("output.png")
# Mostrarlo.

plt.show()


def fn_perceptron(W, x):
    return (-W[1]/W[2])*x+(-W[0]/W[2])


ppn = Perceptron()

W = ppn.fit(TP3_1, y)

generate_img(list_x, list_y, y, fn_perceptron, W, "perceptron-output",
             title="Perceptron con datos clasificados originalmente")


# PUNTO C, mal clasificados
# Puntos bien clasificados -> [ 1  1 -1  1  1 -1  1  1  1  1  1  1 -1 -1  1  1  1  1 -1  1]

# Clase mal clasificada
y1 = [-1,  1, -1,  1,  1, -1,  -1,  -1,  1,
      1,  1,  1, 1, -1,  1,  -1,  1,  1, -1,  -1]
# TP3_2
W1 = ppn.fit(TP3_1, y1)
generate_img(list_x, list_y, y1, fn_perceptron, W1,
             "perceptron-output-punto-c", title="Perceptron con datos mal clasificados")

# SVM TP3_1
clf = svm.SVC()
clf.fit(TP3_withouth_x0, y)

result_TP3 =clf.predict(TP3_withouth_x0)
print("CLASE ORIGNAL: ", y)
print("RESULTADO SVM DATOS ORIGINALES: ", clf.predict(TP3_withouth_x0))
## ¿El hiperplano de separacion es optimo? No. La recta  tiene que tener a la misma distancia de los puntos de las dos clases
# SVM TP3_2
clf2 = svm.SVC()
clf2.fit(TP3_withouth_x0, y1)
result_TP3_2 = clf2.predict(TP3_withouth_x0)
print("CLASE MAL CLASIFICADA: ", y1)
print("RESULTADO SVM DATOS MAL CLASIFICADOS: ", result_TP3_2)


## Comparar resultados punto c con punto a



confusion_matrix = metrics.confusion_matrix(y, result_TP3 )
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.title("CLASE VERDADERA Y RESULTADO")
plt.ylabel('Clase resultado')
plt.xlabel('Clase verdadera')
plt.show()

confusion_matrix = metrics.confusion_matrix(y, result_TP3_2 )
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cm_display.plot()
plt.title("CLASE VERDADERA DE DATOS MAL CLASIFICADOS Y RESULTADO")
plt.ylabel('Clase resultado')
plt.xlabel('Clase verdadera')
plt.show()
############################# PUNTO 2 ##############################################


# Clase cielo: 0 pasto: 1 vaca: 2
examples_class = [0,1,2]
examples = ["cielo", "pasto", "vaca"]





# i -> clase
images_with_class = []

images = ([cv.imread(img_name+".jpg") for i, img_name in enumerate(examples)])

for i, img in enumerate(images):
    row, column, _ = img.shape
    _range = 4000
    for irow in range(_range):
        pixel_class = []
        for color in img[random.randint(0, row-1), random.randint(0, column-1), :].tolist():
            pixel_class.append(color)
        pixel_class.append(i)
        images_with_class.append(pixel_class)


def get_random_elements(data, n):
    random_elements = []
    for i in range(n):
        random_index = random.randint(0, len(data)-1)
        random_elements.append(data[random_index])
    return random_elements


def split_class(data):

    class_data = list()
    X = []
    for ele in data:
        class_data.append(ele[len(ele)-1])
        X.append(ele[:-1])

    return X, class_data


training_set = get_random_elements(images_with_class, 800)
test_set = get_random_elements(images_with_class, 500)
# print("IMAGES with class",images_with_class )
Xtest, ytest = split_class(test_set)
Xtraining_set, ytraining_set = split_class(training_set)


# Entrenando con kernel lineal
clflinear = svm.SVC(kernel="linear")
# print("Ytest",ytraining_set)
clflinear.fit(Xtraining_set, ytraining_set)

result_linear_test = clflinear.predict(Xtest)
# print("LINEAR TEST", result_linear_test)
# Entrenando con kernel polinomial

clfpoly = svm.SVC(kernel="poly")
clfpoly.fit(Xtraining_set, ytraining_set)
result_polinomial_test = clfpoly.predict(Xtest)
# print("POLINOMIAL TEST", result_polinomial_test)

# Entrenando con kernel rdf
clfrbf = svm.SVC(kernel="rbf")
clfrbf.fit(Xtraining_set, ytraining_set)
result_rbf_test = clfrbf.predict(Xtest)
# print("RBF TEST", result_rbf_test)

cow_1 = cv.imread("cow_1.jpg")
# Clase cielo: 0 pasto: 1 vaca: 2


def make_img(image, clf):
    row, column, _ = image.shape
    for r_index in range(row):
        for c_index in range(column):
            result, *_ = clf.predict([image[r_index, c_index]])
            image[r_index, c_index, 0] = 0
            image[r_index, c_index, 1] = 0
            image[r_index, c_index, 2] = 0
            image[r_index, c_index, result] = 255
    cv.imshow("Imagen", image)
    cv.waitKey(0)


make_img(cow_1,clflinear)
# make_img(cow_1,clfpoly)
# make_img(cow_1,clfrbf)
cow_test = cv.imread("cow-test-2.jpg")
#print("COW", cow_test)
# make_img(cow_test,clflinear)
# make_img(cow_test,clfpoly)
# make_img(cow_test,clfrbf)