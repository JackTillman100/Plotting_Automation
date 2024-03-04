## File name: fitness_score_sorting.py
## Authors: Jack Tillman (tillman.100@osu.edu), Josh Lahr (lahr.23@osu.edu)
## Date: 12/06/2021
## Purpose:
##     This script is designed to read the fitness scores of all of the individuals in every generation
##     after the genetic algorithm (the loop) is ran and output the top 5 individuals along with their fitness 
##     scores
##
## Instructions:
##              To run, give the following arguments:
##              directory of generation data
##                      
##
## Example:
##              python fitness_score_sorting.py generation_data_directory
##                      This will find the top 5 antenna once the loop is finished running


### Appends all of the fitness scores to a list
##      First open fitness scores for 1 generation and append that to its own list
##      Append that list onto the main list so that the main list has the following indexes:
##                          fitnessScores[genNumber][individualNumber]

##Sort through the main list to find the top 5 fitness scores

##Print the fitness scores and the individual they're associated with in the following format:
##      Individual: Fitness Score
##      Ex. 09_15: 5.36201

##################################################################################
######## Puts all fitness scores in a dictionary organized by generations ########
##################################################################################

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
            
all_fitness_scores = []
all_indices = []
all_gen = []


organized_fitness_scores = {}

for gen in range(0, g.num_gens - 1):

    fitness_scores=[]
    with open(g.source  + '/' + "Generation_" + str(gen) + '/' + str(gen) + "_vEffectives.csv") as f:
        for row in f:
            if row[0].isdigit():
                
                row = row.split(',')
                
                ### Each fitness score is appended to a list
                fitness_scores.append(float(row[0]))
    
    organized_fitness_scores[gen] = fitness_scores
            
print('Test:')
print('=================================================================')
print()
print(organized_fitness_scores)



