#This file leverages gaussian estimation from a likelihood perspective
#It also goes to on to prepare a naive bayes classifier using numpy
#It doesn't leverages scikit as the intent here is to understand in deep how each step works
#Data set for learning iris.data
#Data set for predicting bezdekIris.data

import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal


SETOSA = "Iris-setosa"
VERSI = "Iris-versicolor"
VIRGI = "Iris-virginica"



def parse_line(lst):
    sepal_length = float(lst[0])
    sepal_width = float(lst[1])
    petal_length = float(lst[2])
    petal_width = float(lst[3])
    result_class = str(lst[4]).replace('\n','')
    arr = np.array([sepal_length,sepal_width,petal_length,petal_width])
    return (arr,result_class)


def likelihood(samples):
    mean = np.mean(samples,axis=0)
    variance = np.var(samples,axis=0)
    #print("var " + str(variance))
    #print(variance.shape)
    s = variance.shape[0]
    cov = np.zeros((s,s))
    #print(cov)
    i = 0
    for i in range(0,s):
        cov[i][i] = variance[i]
    #print(cov)
    return mean,cov


def guess_uniform(sample,class_mean_vars):
    selected_class = SETOSA
    diff_setosa = np.linalg.norm(sample-class_mean_vars[0][0])
    diff = diff_setosa
    diff_versi = np.linalg.norm(sample-class_mean_vars[1][0])
    if (diff_versi < diff):
        selected_class = VERSI
        diff = diff_versi
    diff_virg = np.linalg.norm(sample-class_mean_vars[2][0])
    if (diff_virg < diff):
        selected_class = VIRGI
        diff = diff_virg

    return selected_class,diff

def guess_gaussian_bayes(sample,class_mean_vars,priors):
    selected_class = SETOSA
    selected_pr = 0.0
    setosa_mean_var = class_mean_vars[0]
    setosa_pr = multivariate_normal.pdf(sample,setosa_mean_var[0],setosa_mean_var[1])*priors[0]
    selected_pr = setosa_pr
    versi_mean_var = class_mean_vars[1]
    versi_pr = multivariate_normal.pdf(sample,versi_mean_var[0],versi_mean_var[1])*priors[1]
    if (versi_pr > selected_pr):
        selected_class = VERSI
        selected_pr = versi_pr
    virg_mean_var = class_mean_vars[2]
    virg_pr = multivariate_normal.pdf(sample,virg_mean_var[0],virg_mean_var[1])*priors[2]
    if (virg_pr > selected_pr):
        selected_class = VIRGI
        selected_pr = virg_pr
    
    return selected_class,selected_pr
    
    
def read_file(file_name):
    f = open(file_name,"r")
    samples_setosa = None
    setosa_set = False
    samples_versicolor = None
    versi_set = False
    samples_virginica = None
    virgi_set = False
    for line in f:
        lst = line.split(",")
        sample,name = parse_line(lst)
        if name == "Iris-setosa":
            if setosa_set == False:
                samples_setosa = sample
                setosa_set = True
            else:
                samples_setosa = np.vstack((samples_setosa,sample))
        if name == "Iris-versicolor":
            if versi_set == False:
                samples_versicolor = sample
                versi_set = True
            else:
                samples_versicolor = np.vstack((samples_versicolor,sample))
        if name == "Iris-virginica":
            if virgi_set == False:
                samples_virginica = sample
                virgi_set = True
            else:
                samples_virginica = np.vstack((samples_virginica,sample))
            #print(str(sample) + " " + str(name))
    
    #print(samples_setosa)
    setosa_mean,setosa_var = likelihood(samples_setosa)
    versicolor_mean,versicolor_var = likelihood(samples_versicolor)
    virginica_mean,virginica_var = likelihood(samples_virginica)
    likelihoods = [(setosa_mean,setosa_var),
                   (versicolor_mean,versicolor_var),
                   (virginica_mean,virginica_var)]

    #print("Setosa shape " + str(samples_setosa.shape[0]))
    total_size = 0.0
    total_size = total_size + samples_setosa.shape[0]
    total_size = total_size + samples_versicolor.shape[0]
    total_size = total_size + samples_virginica.shape[0]

    priors = [samples_setosa.shape[0]/total_size,
              samples_versicolor.shape[0]/total_size,
              samples_virginica.shape[1]/total_size]
    
    return likelihoods,priors

def predict(predict_file_name,likelihoods,priors):
    f = open(predict_file_name,"r")
    samples_setosa = None
    samples_versicolor = None
    samples_virginica = None
    success = 0.0
    failure = 0.0
    for line in f:
        lst = line.split(",")
        sample,name = parse_line(lst)
        #guess_class,dist = guess(sample,likelihoods)
        #print(str(guess_class) + " " + str(name))
        gaussian_class,pr = guess_gaussian_bayes(sample,likelihoods,priors)
        #print(str(gaussian_class) + " " + str(name))
        if (gaussian_class == name):
            success = success + 1.0
        else:
            failure = failure + 1.0

    return success/(success+failure)



likelihoods,priors = read_file("/Users/rasrivastava/neural_net/ML_DATA/iris.data")
success_rate = predict("/Users/rasrivastava/neural_net/ML_DATA/bezdekIris.data",likelihoods,priors)
print(success_rate*100.0)
