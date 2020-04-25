# Genetic_Algorithm_Tuning
Genetic algorithm designed for tuning neural nets.
* Currently, project on hold, but still in progress. Need dataset. (2/5/20)
* The idea is to modify the model architecture across multiple neural networks
  to maximize model accuracy on datasets.

## Genetic Algorithms:
Classical AI inspired by Genetics. 

## Neural Nets:
Machine Learning based on biology, specifically, collections of neurons.

## Set Up:
* `pip install numpy`
* `pip install tensorflow`
* `pip install matplotlib`
* `pip install keras`
* `pip install sklearn`

## Learnings:
* GA does not guarantee optimal solution/global maximum, it finds the local maximums.
    * Similar to a mass local search algorithm
    * Greater population size results in better variation, however takes more time/space complexity
    * Higher mutation rate also results in better variation - 45% ~ 55% pretty good - 70 ish %
    

## Resources:
* Collect Top K 
    - https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
* Matplotlib plots 
    - https://matplotlib.org/tutorials/introductory/pyplot.html
* Rescale Matplotlib 
    - https://stackoverflow.com/questions/10984085/automatically-rescale-ylim-and-xlim-in-matplotlib
* Horizontal stack with numpy 
    - https://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html
* Concatinate numpy 
    - https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html
* Random choice numpy 
    - https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.choice.html
* Keras Datasets documentation
    - https://keras.io/datasets/