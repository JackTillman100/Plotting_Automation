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

