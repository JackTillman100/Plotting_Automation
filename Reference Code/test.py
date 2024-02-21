## File name: image_finder.py
## Authors: Audrey Zinn (zinn.60@osu.edu), Bailey Stephens (stephens.761@osu.edu)
## Date: 12/06/2021
## Purpose:
##     This script is designed to read the fitness scores for the current generation
##     and select the antenna with the best, middle, and worst fitness scores to
##     save an image of
##
## Instructions:
##              To run, give the following arguments
##                      source directory, generation #
##
## Example:
##              python image_finder.py sourceDir 23
##                      This will find the best, middle, and worst individuals
##                      in generation 23 using data in the sourceDir directory.


## Imports
import numpy as np
import argparse
import csv

## Arguments
parser = argparse.ArgumentParser()
parser.add_argument("source", help="Name of source folder from home directory", type=str)
parser.add_argument("num_gens", help="Number of generations (ex:16)", type=int)
parser.add_argument("num_individuals", help="Number of individuals (ex:16)", type=int)
parser.add_argument("filename", help="Name of saved image", type=str)
g=parser.parse_args()

##############
### SWITCH ###
##############

is_single_line=False

##############

## Declare an empty list that will hold the fitness scores 
fitness_scores = [0, 0, 0, 0, 0]
gen_number = [0, 0, 0, 0, 0]
indiv_number = [0, 0, 0, 0, 0]
best_DNA = [[0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]]

### Lists containing data across all generations
all_fitness_scores = []
all_indices = []
all_DNA = []
all_gen = []

###################################################
######         Function Definitions          ######
###################################################

### Function with one return value
def matching_DNA(gen1, indiv1, gen2, indiv2):
    
    """
    Purpose: Determines whether two specified inidividuals have matching DNA
    
    Inputs
    gen1: Generation # of first individual
    indiv1: Individual # of first individual (raw index, not in-gen)
    gen2: Generation # of second individual
    indiv2: Individual # of second individual (raw index, not in-gen)
    
    Outpout
    Boolean indicating whether two individuals are identical
    """
    
    ### Converts raw index into in-generation index
    in_gen_index_1 = int(indiv1%g.num_individuals)
    in_gen_index_2 = int(indiv2%g.num_individuals)
    
    ### Stores DNA for each individual into variables
    DNA1 = all_DNA[gen1][in_gen_index_1]
    DNA2 = all_DNA[gen2][in_gen_index_2]
    
    ### TESTING
    #if DNA1==DNA2:
    #    print(indiv1, gen1)
    #    print(indiv2, gen2)
    #    print()
    
    ### Returns a boolean value indicating whether two individuals are identical
    return DNA1==DNA2

### Function with one return value
def get_gen(indiv_index):
    
    """
    Purpose: 
    """
    
    return int((indiv_index-indiv_index%g.num_individuals)/g.num_individuals)
    
###################################################
######               Main Code               ######
###################################################

print(g.num_gens)

### Iterates over the data from every generation
for gen in range(0, g.num_gens):
    
    ### Temporary list holding fitness scores for a given generation
    fitness_scores=[]
    
    ### First we need to open the csv file that contains the values of the fitness scores.
    with open(g.source  + '/' + "Generation_" + str(gen) + '/' + str(gen) + "_generationDNA.csv") as f:
        
        ### We then itereate over every row in the fitnessScores.csv file
        for row in f:
            
            ### Checks if the current row is a fitness score or not
            if row[0].isdigit():
                
                row = row.split(',')
                
                ### Each fitness score is appended to a list
                fitness_scores.append(float(row[0]))
    
    ### Opens generation data
    with open(g.source  + '/' + "Generation_" + str(gen) + '/' + str(gen) + "_generationDNA.csv") as f:
        lines=f.readlines()
        
        ### Splits data into readable form
        for i in range(len(lines)):
            lines[i] = lines[i].strip()
            lines[i] = lines[i].split(',')
        
        ### Holds DNA for each indiv in generation
        genDNA=[]
        
        ### DNA header line number
        j=9
        
        ### Runs if DNA is expressed in single-line format (old format)
        if is_single_line:
            ### Iterates over each pair of lines after header lines
            while j < len(lines):
                
                #print("YES! YOU DID IT!")
                
                ### Converts DNA pairs into floats
                indivDNA=[float(lines[j][num]) for num in range(len(lines[j]))]
                
                ### Adds DNA to gen list
                genDNA.append(indivDNA)
                
                #print(genDNA)
                
                j+=1
        
        ### Runs if DNA is expressed in double-line format (new format)
        else:
            ### Iterates over each pair of lines after header lines
            while j < len(lines):
                
                ### Converts DNA pairs into floats
                row1DNA=[float(lines[j][num]) for num in range(len(lines[j]))]
                row2DNA=[float(lines[j+1][num]) for num in range(len(lines[j+1]))]
                
                ### Combines pairs of DNA
                indivDNA = row1DNA + row2DNA
                
                ### Adds DNA to gen list
                genDNA.append(indivDNA)
                
                j+=2
    
    ### Makes a list of indices and zips it with the list of fitness scores
    index_list = [i + g.num_individuals * gen for i in range(len(fitness_scores))]
    zipped_fitness = zip(fitness_scores, index_list)
    
    ### Sorts the fitness scores / indices pairs and separates the lists
    sorted_scores = sorted(zipped_fitness)
    tuples = zip(*sorted_scores)
    sorted_fitness_scores, sorted_indices = [list(tuple) for tuple in  tuples]
    #sorted_fitness_scores, sorted_indices, sorted_genDNA = [list(tuple) for tuple in  tuples]
    
    ### Adds data to lists containing sorted scores and indices
    all_fitness_scores.append(sorted_fitness_scores)
    all_indices.append(sorted_indices)
    
    ### Unsorted 3D array containing each individuals DNA
    all_DNA.append(genDNA)

### Long 1D lists containing sorted data
big_fitness = []
big_indices = []
test_DNA_1D = []

### Fills 1D lists containing sorted data
for i in range(len(all_fitness_scores)):
    big_fitness += all_fitness_scores[i]
    big_indices += all_indices[i]
    test_DNA_1D += all_DNA[i]

### Sorts 1D lists across all generations
zipped_big = zip(big_fitness, big_indices)
big_sorted_scores = sorted(zipped_big)
tuples = zip(*big_sorted_scores)
big_sorted_fitness, big_sorted_indices = [list(tuple) for tuple in  tuples]

### Iterating variable for moving backwards through our data
indiv_checker=-2

### Number of unique individuals desired from output
#n_unique=g.num_individuals*g.num_gens
n_unique=5

### Boolean flag determining whether n-unique individuals have been found
n_cap_met = False

### Lists containing individual indices and fitness scores of top n-unique individuals
### NOTE: Initialized with best individual from entire run
best_n_indivs=[big_sorted_indices[-1]]
best_n_scores=[big_sorted_fitness[-1]]

### Variable counting how many individuals have matching DNA
num_matching=0

### Runs until top n unique individuals are found
while not n_cap_met and np.abs(indiv_checker) <= len(big_sorted_indices):
    
    #print(indiv_checker)
    last_unique_index = best_n_indivs[-1]
    new_indiv = big_sorted_indices[indiv_checker]
    
    #print(big_sorted_fitness[indiv_checker] - best_n_scores[-1])
    #print(big_sorted_fitness[indiv_checker], best_n_scores[-1])
    
    matching=False
    for unique_detector in best_n_indivs:
        if matching_DNA(get_gen(new_indiv), new_indiv, get_gen(unique_detector), unique_detector):
            #num_matching+=1
            matching=True
    
    if matching:
        #print("FOUND ONE", indiv_checker, num_matching)
        num_matching+=1
    
    if not matching:
        
        #unique_per_gen[get_gen(new_indiv)]+=1
        
        best_n_indivs.append(new_indiv)
        best_n_scores.append(big_sorted_fitness[indiv_checker])
    
    ### Stops the loop after n best individuals have been found
    if len(best_n_indivs)==n_unique:
        n_cap_met=True
    
    ### Iterates through best-performing individuals
    indiv_checker-=1

#print("AND OUR WINNERS ARE....")

print(best_n_scores)
print()
for index in best_n_indivs:
    
    print("DNA:", all_DNA[get_gen(index)][index%g.num_individuals])
    print("GEN:", get_gen(index))
    print("INDIVIDUAL:", index%g.num_individuals)
    print()

print(best_n_indivs)
print(best_n_scores)
#print("REPEATED INDIVS:", num_matching)
#print(test_DNA_1D)
#print(len(np.unique(test_DNA_1D)))
#print("UNIQUE:", len(set(map(tuple, test_DNA_1D))))

### List of each generation number in run
xvals=[i for i in range(0, g.num_gens+1)]

### List containing number of unique individuals per generation (initialized to zeroes)
unique_per_gen = [0 for i in range(0, g.num_gens+1)]

### List containing DNA already analyzed for future reference
analyzed_DNA=[]

### Iterates over every individual's DNA from a generation
for i in range(len(test_DNA_1D)):
    
    ### Runs if current individual's DNA has appeared before
    if test_DNA_1D[i] in analyzed_DNA:
        pass
    
    ### Runs if individuals's DNA is unique
    else:
        unique_per_gen[get_gen(i)]+=1
        
    ### Adds individuals DNA to a list (regardless of whether it was or was not unique)
    analyzed_DNA.append(test_DNA_1D[i])


#print(unique_per_gen)

### Filename extension
use_this_one = g.filename + "_Unique_Plot"

import matplotlib.pyplot as plt
plt.scatter(xvals, unique_per_gen, label='# Unique Individuals')
plt.legend()
#plt.ylim([0, g.num_individuals])
plt.xlabel('generation')
plt.ylabel('unique # of individuals')
plt.savefig(use_this_one)
