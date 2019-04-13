"""
DISCENTES: Guilherme Afonso R. Gomes

=======================================================
Exemplo: Reconhecimento Facial usando eigenfaces e SVMs
=======================================================

O conjunto de dados(dataset) usado nesse exemplo é uma base
pré-processada do "Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

.. _LFW: http://vis-www.cs.umass.edu/lfw/

Resultados esperados para as 3 pessoas mais representativas
no conjunto de dados:

================== ============ ======= ========== =======
                     precision   recall  f2-score  support
================== ============ ======= ========== =======
     Ariel Sharon       0.67      0.92      0.77        13
     Colin Powell       0.75      0.78      0.76        60
  Donald Rumsfeld       0.78      0.67      0.72        27
    George W Bush       0.86      0.86      0.86       146
Gerhard Schroeder       0.76      0.76      0.76        25
      Hugo Chavez       0.67      0.67      0.67        15
       Tony Blair       0.81      0.69      0.75        36

      avg / total       0.80      0.80      0.80       322
================== ============ ======= ========== =======

"""
from __future__ import print_function
from sklearn import metrics
from time import time
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


print(__doc__)
# mostrando progresso na saída
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# #############################################################################
# Download dos dados, caso eles não estejam no disco e carregando eles como numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=143, resize=0.4)

# introspecção dos arrays de imagens para achar os formatos(para plotar)
n_samples, h, w = lfw_people.images.shape

# para aprendizado de máquina usa-se 2 dados diretamente(como informações relativas
# de posições são ignoradas por esse modelo)
X = lfw_people.data
n_features = X.shape[1]

# o rótulo para prever é o id da pessoa
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("\nTamanho total do conjunto de dados(Total dataset size):")
print("numero de amostras(n_samples): %d" % n_samples)
print("n_características(n_features): %d" % n_features)
print("n_classes: %d" % n_classes)

# #############################################################################
# Divide em um conjunto de treino e um conjunto de teste usando stratified k fold

# Dividindo em um conjunto de treino e e um conjunto de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=10)

# #############################################################################
# Calcula o PCA(eigenfaces) no conjunto de dados das faces (tratado como conjunto de dados não rotulado):
# sem supervisão de característia de extração / redução de dimensionalidade
n_components = 150
print("\nExtraindo o top %d eigenfaces de %d faces" % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
print("Feito em %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))
print("\nProjetando a entrada de dados na base ortonormal do eigenfaes")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("Feito em %0.3fs" % (time() - t0))
# #############################################################################
# Treinando um modelo de classificação SVM

print("\nAjustando o classificador ao conjunto de treino com as  função de kernel RBF :")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
# kernel -> RBF
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, cv=5)
clf = clf.fit(X_train_pca, y_train)
print("Feito em %0.3fs" % (time() - t0))
print("\nMelhor estimador achado pela pesquisa:")
print("Kernel = RBF")
print(clf.best_estimator_)

print("\nAjustando o classificador ao conjunto de treino com as  funções de kernel Linear:")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
# kernel -> linear
classifier = GridSearchCV(SVC(kernel='linear', class_weight='balanced'), param_grid, cv=5)
classifier = classifier.fit(X_train_pca, y_train)
print("Feito em %0.3fs" % (time() - t0))
print("\nMelhor estimador achado pela pesquisa:")
print("Kernel = linear")
print(classifier.best_estimator_)
# #############################################################################
# Avaliação quantitativa do modelo de qualidade no conjunto de teste

print("\nPredição dos nomes das pessoas no conjunto de teste")
t0 = time()
# iniciando vetor com os valores de predição para RBF
y_pred_rbf = clf.predict(X_test_pca)
# iniciando vetor com os valores de predição para linear
y_pred_linear = classifier.predict(X_test_pca)
print("Feito  em %0.3fs" % (time() - t0))

# Matriz de confusão RBF
print("\nRBF:")
print(classification_report(y_test, y_pred_rbf, target_names=target_names))
print(confusion_matrix(y_test, y_pred_rbf, labels=range(n_classes)))

# Matriz de confusão Linear
print("\nLinear:")
print(classification_report(y_test, y_pred_linear, target_names=target_names))
print(confusion_matrix(y_test, y_pred_linear, labels=range(n_classes)))

# Iniciando Curva ROC para os kernels rbf e linear
fpr_lin = dict()
fpr_rbf = dict()
tpr_lin = dict()
tpr_rbf = dict()
roc_auc_lin = dict()
roc_auc_rbf = dict()
# score Linear e RBF com a função decisão passando o conjunto de teste reduzido em sua dimensão
y_score_rbf = clf.decision_function(X_test_pca)
y_score_linear = classifier.decision_function(X_test_pca)
# aplicando ao conjunto de teste y_test a função label_binarize
y_test_roc_curve = label_binarize(y_test, classes=[0, 1, 2])

# RBF e Linearfor
for i in range(n_classes):
    fpr_rbf[i], tpr_rbf[i], _ = roc_curve(y_test_roc_curve[:, i], y_score_rbf[:, i])
    fpr_lin[i], tpr_lin[i], _ = roc_curve(y_test_roc_curve[:, i], y_score_linear[:, i])
    roc_auc_lin[i] = auc(fpr_lin[i], tpr_lin[i])
    roc_auc_rbf[i] = auc(fpr_rbf[i], tpr_rbf[i])
# Compute micro-average ROC curve and ROC area
fpr_rbf["micro"], tpr_rbf["micro"], _ = roc_curve(y_test_roc_curve.ravel(), y_score_rbf.ravel())
roc_auc_rbf["micro"] = auc(fpr_rbf["micro"], tpr_rbf["micro"])
# Compute micro-average ROC curve and ROC area
fpr_lin["micro"], tpr_lin["micro"], _ = roc_curve(y_test_roc_curve.ravel(), y_score_linear.ravel())
roc_auc_lin["micro"] = auc(fpr_lin["micro"], tpr_lin["micro"])
# plotar curva roc
plt.figure()
lw = 3
plt.plot(fpr_rbf[2], tpr_rbf[2], color='blue', lw=lw, label='RBF - Curva ROC (area = %0.5f)' % roc_auc_rbf[2])
plt.plot(fpr_lin[2], tpr_lin[2], color='darkorange', lw=lw, label='Linear - Curva ROC (area = %0.5f)' % roc_auc_lin[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Taxa Falso Positivo')
plt.ylabel('Taxa Verdadeiro Positivo')
plt.title('Curva ROC - Receiver Operating Characteristic')
plt.legend(loc="lower right")
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Função auxiliar para plotar Figura"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
# plotar o resultado da predição em uma porção do conjunto de teste para:
# RBF:
def title_RBF(y_pred_rbf, y_test, target_names, i):
    pred_name = target_names[y_pred_rbf[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'RBFpredição: %s\nverdadeiro: %s' % (pred_name, true_name)
prediction_titles_rbf = [title_RBF(y_pred_rbf, y_test, target_names, i)
                     for i in range(y_pred_rbf.shape[0])]
# Linear:
def title_linear(y_pred_linear, y_test, target_names, i):
    pred_name = target_names[y_pred_linear[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'LinPredição: %s\nverdadeiro: %s' % (pred_name, true_name)
prediction_titles_linear = [title_linear(y_pred_linear, y_test, target_names, i)
                     for i in range(y_pred_linear.shape[0])]

plot_gallery(X_test, prediction_titles_rbf, h, w)
plot_gallery(X_test, prediction_titles_linear, h, w)

# plotar a galeria dos eigenfaces mais significativos
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
plt.show()