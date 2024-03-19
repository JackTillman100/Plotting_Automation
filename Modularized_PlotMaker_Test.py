#!/usr/bin/env python3

######################################
#    AraSim simple event reader      #
#        Dennis H. calderon          #
#    calderon-madera.1@osu.edu       #
######################################

######################################
#    Modularized AraSim reader       #
#        Jack Tillman                #
#      tillman.100@osu.edu           #
######################################

#######################################################
"""
=======================
##project_test.py##
======================
Author: Dennis H. Calderon
Email: calderon-madera.1@osu.edu
Date: November 02, 2021
Modified: March 24, 2022
=======================
Descripiton: 
This PYTHON script takes two sets of AraSim output .root files. For each set, it makes a cut for triggered events, pulls variables, and makes histograms comparing the two. 

This script was make for a comparison of antennas for the ARA Bicone (vpol) and an evolved antenna using GENETIS (vpol). This current verion is comparing Direct & Refracted/Reflected Events using variables (theta_rec, rec_ang, reflect_ang) for each simulation run.
Be sure to use the v3.0.0 version of /cvmfs/ara.opensciencegrid.org/v3.0.0/centos7/source/AraSim/ rather than the trunk version which was previously used

=======================
Usage:
python project.py <source> [options] <source_2>
<source_1> is where the ROOT file from your AraSim output
<source_2> is path where the other ROOT file to compare
<source_3> is path where the other ROOT file to compare
<source_4> is path where the other ROOT file to compare
<source_5> is path where the other ROOT file to compare
<source_6> is path where the other ROOT file to compare.
=======================
Options:
[-s2, -s3, -s4, -s5, -s6]  tells program that you are putting in anoter source of simulation files.
=======================
example:
python all_vars.py ../output_files/AraOut.Bicone.run{0..9}.root -s2 ../output_files/AraOut.GENETIS.run{0..9}.root
=======================
"""

#######################################################
import timeit
start = timeit.default_timer()
#######################################################
print("\n")
print('\033[1;37m#\033[0;0m'*50)
print("Now running \033[1;4;5;31mModularized_PlotMaker_Test.py\033[0;0m!")
print('\033[1;37m#\033[0;0m'*50)
print('\n')
##########################################
print("\033[1;37mPlease wait patiently...\033[0;0m")
print('Importing libraries...')

##########################################
#System libraries
#import sys
import argparse
#import csv
#import types
#import os
import warnings
warnings.filterwarnings("ignore")
print('...')

#PyRoot libraries
import ROOT
#from ROOT import TCanvas, TGraph
#from ROOT import gROOT
from ROOT import gInterpreter, gSystem
#from ROOT import TChain, TSelector, TTree
from ROOT import TChain
print('...')

#Python libraries
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import os 
print('...')

#Plotting functions
import modularized_plotting_functions_test as plotFunctions
print('...')

##########################################

########
#Note: Only use AraSim files generated with the v3.0.0 version from the following location: /cvmfs/ara.opensciencegrid.org/v3.0.0/centos7/source/AraSim/'
#We've previously been using the trunk version rather than the v3.0.0 version!
########

#####
#AraSim specific headers needed
gInterpreter.ProcessLine('#include "/cvmfs/ara.opensciencegrid.org/trunk/centos7/source/AraSim/Position.h"')#"/users/PAS0654/dcalderon/AraSim/Position.h"')
gInterpreter.ProcessLine('#include "/cvmfs/ara.opensciencegrid.org/trunk/centos7/source/AraSim/Report.h"')#"/users/PAS0654/dcalderon/AraSim/Report.h"')
gInterpreter.ProcessLine('#include "/cvmfs/ara.opensciencegrid.org/trunk/centos7/source/AraSim/Detector.h"')#"/users/PAS0654/dcalderon/AraSim/Detector.h"')
gInterpreter.ProcessLine('#include "/cvmfs/ara.opensciencegrid.org/trunk/centos7/source/AraSim/Settings.h"')#"/users/PAS0654/dcalderon/AraSim/Settings.h"')

gSystem.Load('/cvmfs/ara.opensciencegrid.org/trunk/centos7/ara_build/lib/libAra.so')#'/users/PAS0654/dcalderon/AraSim/libAra.so') 

##########################################
# We want to give an output file as an input. This checks that we have a fle to read
parser = argparse.ArgumentParser(
        description='Read AraSim file and produce some plots. Can also compare two AraSim files.')
parser.add_argument("source_1", help = "Path to the AraSim file you want to use.", nargs='+')
parser.add_argument("--source_2", "-s2", help = "Path to another AraSim file you want to comprare to.", nargs='+')
parser.add_argument("--source_3", "-s3", help = "Path to another AraSim file you want to comprare to.", nargs='+')
parser.add_argument("--source_4", "-s4", help = "Path to another AraSim file you want to comprare to.", nargs='+')
parser.add_argument("--source_5", "-s5", help = "Path to another AraSim file you want to comprare to.", nargs='+')
parser.add_argument("--source_6", "-s6", help = "Path to another AraSim file you want to comprare to.", nargs='+')

g = parser.parse_args()

##########################################
'''
can put this inside as well
'''

#Using our prewritten function to initialize source_dict & source_names based off of the parsed data
source_dict, source_names = plotFunctions.parsed_data_list(g)

########################
##Variables needed
########################
energy = np.power(10,18)
earth_depth = 6359632.4
core_x = 10000.0
core_y = 10000.0

##################################
###Loop over Events
##################################

print('#'*50)
print("Now lets do the loop")
print("Please wait patiently...")
print('...')
print('\n')

data_dict = {}
var_dict, data_dict, source_names = plotFunctions.data_analysis(source_dict, source_names)
print('\n')
print("We have now looped over all events and selected only triggered events")
print("Now we can let the fun begin...")
print('#'*50)
print('\n')

#######################################
###Plots
#######################################
print('#'*50)
print("Now lets make some plots!")
print('#'*50)

#Setting up plotting variables 
w = 2.0
binsize = np.linspace(-1.0, 1.0, 41)
bindepth = 20
bindistance = np.linspace(0,4000, 21)

bin_cos = np.linspace(-1.0, 1.0, 41)
bin_dist = np.linspace(0,4000, 41)
binsize = np.linspace(-1.0, 1.0, 41)
bindepth = 20
bindistance = np.linspace(0,4000, 41)
fontsize = 12

##Setting up legends 
colors = ['r','b','g','c','m','y']

custom_lines_style = [Line2D([0], [0], color='k', ls='-'),
                      Line2D([0], [0], color='k', ls='--')]

#Making legends
custom_lines_color = []
for i in range(len(source_names)):
        custom_lines_color.append(Line2D([0], [0], color=colors[i], lw=4))
#custom_lines_color.append(Line2D([0], [0], color='k', ls ='-'))
#custom_lines_color.append(Line2D([0], [0], color='k', ls ='--'))

legend_names = list(data_dict.keys())
#legend_names.append('Direct')
#legend_names.append('Refracted')

custom_legend = []
for i in range(len(source_names)):
        custom_legend.append(Line2D([0], [0], color=colors[i], lw=4))

#If makelabel = 1 a legend is generated, if it is 0, a legend is not
makelabel = 1

#Variable arrays for plotting
hist_vars = ['rec_ang','theta_rec','view_ang','launch_ang','reflect_ang',
             'nnu_theta', 'nnu_phi',
             'dist', 'ShowerEnergy', 'depth', 'distance', 'flavor', 'elast', 'weight']
bins = [bin_cos, bin_cos, bin_cos, bin_cos, bin_cos, bindistance]
ang_strings = ['ang', 'theta', 'phi']

#To plot the collected data, we will call functions from a prewritten python file

##################
#####Plotting#####
##################

'''
In the current paper, the direct events from the theta_rec histogram plot are used
'''

print('\n')
print("Histograms!")
print('\n')
print("All at once!")
for j in range(len(hist_vars)):
        print("Plotting...")
        plt.figure(j, figsize=(8,6))
        for i in range(len(source_names)):
                plotFunctions.hist_maker(data_dict, bin_cos, bindistance, hist_vars[j], source_names[i], colors[i], fontsize, makelabel, custom_lines_color, legend_names)
                plt.title("{0}".format(data_dict[source_names[i]]['Total_Events']))
        plt.savefig('test_plots/Hist_{0}_All.png'.format(hist_vars[j]),dpi=300)
        plt.clf()

print('\n')
print("Bicone vs. Rest...")
for j in range(len(hist_vars)):
        print("Plotting...")
        plt.figure(j, figsize=(8,6))
        for i in range(1, len(source_names)):
                temp_legend_names = [legend_names[0], legend_names[i]]
                temp_legend_colors = [custom_lines_color[0], custom_lines_color[i]]
                plotFunctions.hist_maker(data_dict, bin_cos, bindistance, hist_vars[j], source_names[0], colors[0], fontsize, makelabel, temp_legend_colors, temp_legend_names)
                plotFunctions.hist_maker(data_dict, bin_cos, bindistance, hist_vars[j], source_names[i], colors[i], fontsize, makelabel, temp_legend_colors, temp_legend_names)
                plt.title("{0}".format(data_dict[source_names[i]]['Total_Events']))
                plt.savefig('test_plots/Hist_{0}_{1}_{2}.png'.format(hist_vars[j],source_names[0],source_names[i]),dpi=300)
                plt.clf()
print('Done!')
print('\n')

print("Scatter Plots!")

scatter_vars = ['distance', 'depth', 'dist_0', 'rec_ang_0', 'theta_rec_0']

for i in range(len(source_names)):
        print("Plotting...")
        plt.figure(i, figsize=(8,6))
        temp_legend_names = [legend_names[i]]
        temp_legend_colors = [custom_lines_color[i]]
        plotFunctions.scatter_maker(scatter_vars[0], scatter_vars[1], data_dict, bin_cos, bindistance, source_names[i], colors[i], fontsize, makelabel, temp_legend_colors, temp_legend_names)
        plt.title("{0}".format(data_dict[source_names[i]]['Total_Events']))
        plt.savefig('test_plots/Scatter_{2}_{0}_{1}_.png'.format(scatter_vars[0], 
                                                                 scatter_vars[1], 
                                                                 source_names[i], colors[i]), dpi=300)
        plt.clf()
print("Done!")
print('\n')
 
print("2D Histogram Plots!")
for i in range(len(source_names)):
        print("Plotting...")
        makelabel = 0
        plt.figure(i, figsize=(8,6))
        plotFunctions.multi_hist(scatter_vars[2], scatter_vars[4], data_dict, bin_cos, bindistance, bin_dist, source_names[i], fontsize, makelabel, custom_lines_color, legend_names)
        plt.savefig('test_plots/2DHist_{2}_{0}_{1}_.png'.format(scatter_vars[2], 
                                                                scatter_vars[3], 
                                                                source_names[i]), dpi=300)
        plt.clf()
print("Done!")
print('\n')

print("2D Histogram Comparison Plots!")

for i in range(1, len(source_names)):
        print("Plotting...")
        makelabel = 0
        plt.figure(i, figsize=(8,6))
        plotFunctions.diff_hist(scatter_vars[2], scatter_vars[4], source_names[0], source_names[i], source_names, bin_dist, bin_cos, source_names, data_dict, 12, makelabel, custom_lines_color, legend_names)
        plt.savefig('test_plots/2DHistDiff_{2}_{3}_{0}_{1}_.png'.format(scatter_vars[2], 
                                                                        scatter_vars[3], 
                                                                        source_names[0], 
                                                                        source_names[i]), dpi=300)
        plt.clf()
print("Done!")
print('\n')

hist_vars = ['rec_ang','theta_rec','view_ang','launch_ang','reflect_ang',
             'nnu_theta', 'nnu_phi',
             'dist', 'ShowerEnergy', 'depth', 'distance', 'flavor', 'elast', 'weight']


print("PDF of Histograms!")
#making pdfs of all histogram
plt.figure(1001, figsize=(8.5,11))
plt.suptitle('All sources', fontsize=16)
makelabel = 1
for i in range(len(source_names)):
        print("Plotting...")
        plt.subplot(3, 2, 1)
        plotFunctions.hist_maker(data_dict, bin_cos, bindistance, 'nnu_theta', source_names[i], colors[i], 8, makelabel, custom_lines_color, legend_names)
        plt.subplot(3, 2, 2)
        plotFunctions.hist_maker(data_dict, bin_cos, bindistance, 'theta_rec', source_names[i], colors[i], 8, makelabel, custom_lines_color, legend_names)
        plt.subplot(3, 2, 3)
        plotFunctions.hist_maker(data_dict, bin_cos, bindistance, 'ShowerEnergy', source_names[i], colors[i], 8, makelabel, custom_lines_color, legend_names)
        plt.subplot(3, 2, 4)
        plotFunctions.hist_maker(data_dict, bin_cos, bindistance, 'weight', source_names[i], colors[i], 8, makelabel, custom_lines_color, legend_names)
        plt.subplot(3, 2, 5)
        plotFunctions.hist_maker(data_dict, bin_cos, bindistance, 'dist', source_names[i], colors[i], 8, makelabel, custom_lines_color, legend_names)
        plt.subplot(3, 2, 6)
        plotFunctions.hist_maker(data_dict, bin_cos, bindistance, 'depth', source_names[i], colors[i], 8, makelabel, custom_lines_color, legend_names)
plt.savefig('test_plots/All_Sources_Histograms.pdf', dpi=300)
plt.clf()
print("Done!")
print('\n')


print("More PDFs of Histograms!")
for i in range(len(source_names)):
        print("Plotting...")
        makelabel = 1
        temp_legend_names = [legend_names[0], legend_names[i]]
        temp_legend_colors = [custom_lines_color[0], custom_lines_color[i]]
        plt.figure(1001, figsize=(8.5,11))
        supTitle = '{0} and {1}'.format(source_names[0],source_names[i])
        plt.suptitle(supTitle, fontsize = 16)

        plt.subplot(3, 2, 1)
        plotFunctions.hist_maker(data_dict, bin_cos, bindistance, 'nnu_theta', source_names[0], colors[0], 8, makelabel, temp_legend_colors, temp_legend_names)
        plotFunctions.hist_maker(data_dict, bin_cos, bindistance, 'nnu_theta', source_names[i], colors[i], 8, makelabel, temp_legend_colors, temp_legend_names)

        plt.subplot(3, 2, 2)
        plotFunctions.hist_maker(data_dict, bin_cos, bindistance, 'theta_rec', source_names[0], colors[0], 8, makelabel, temp_legend_colors, temp_legend_names)
        plotFunctions.hist_maker(data_dict, bin_cos, bindistance, 'theta_rec', source_names[i], colors[i], 8, makelabel, temp_legend_colors, temp_legend_names)

        plt.subplot(3, 2, 3)
        plotFunctions.hist_maker(data_dict, bin_cos, bindistance, 'ShowerEnergy', source_names[0], colors[0], 8, makelabel, temp_legend_colors, temp_legend_names)
        plotFunctions.hist_maker(data_dict, bin_cos, bindistance, 'ShowerEnergy', source_names[i], colors[i], 8, makelabel, temp_legend_colors, temp_legend_names)

        plt.subplot(3, 2, 4)
        plotFunctions.hist_maker(data_dict, bin_cos, bindistance, 'weight', source_names[0], colors[0], 8, makelabel, temp_legend_colors, temp_legend_names)
        plotFunctions.hist_maker(data_dict, bin_cos, bindistance, 'weight', source_names[i], colors[i], 8, makelabel, temp_legend_colors, temp_legend_names)

        plt.subplot(3, 2, 5)
        plotFunctions.hist_maker(data_dict, bin_cos, bindistance, 'dist', source_names[0], colors[0], 8, makelabel, temp_legend_colors, temp_legend_names)
        plotFunctions.hist_maker(data_dict, bin_cos, bindistance, 'dist', source_names[i], colors[i], 8, makelabel, temp_legend_colors, temp_legend_names)

        plt.subplot(3, 2, 6)
        plotFunctions.hist_maker(data_dict, bin_cos, bindistance, 'depth', source_names[0], colors[0], 8, makelabel, temp_legend_colors, temp_legend_names)
        plotFunctions.hist_maker(data_dict, bin_cos, bindistance, 'depth', source_names[i], colors[i], 8, makelabel, temp_legend_colors, temp_legend_names)

        plt.savefig('test_plots/Histograms_{0}_{1}.pdf'.format(source_names[0],source_names[i], dpi=300))
        plt.clf()

print("Done!")
print('\n')



#Doing it for all in a for loop
print("PDF of Scatter Plots, 2D Histograms, and Comparison 2D Histograms!")
for i in range(1, len(source_names)):
        print("Plotting...")
        plt.figure(20001, figsize=(8.5,11))
        supTitle = '{0} and {1}'.format(source_names[0], source_names[i])
        temp_legend_names = [legend_names[0], legend_names[i]]
        temp_legend_colors = [custom_lines_color[0], custom_lines_color[i]]
        makelabel = 0
        plt.suptitle(supTitle, fontsize = 16)

        plt.subplot(3,2,1)
        plotFunctions.scatter_maker('dist_0', 'theta_rec_0', data_dict, bin_cos, bindistance, source_names[0], colors[0], 8, makelabel, temp_legend_colors, temp_legend_names)

        plt.subplot(3, 2, 3)
        plotFunctions.multi_hist('dist_0', 'theta_rec_0', data_dict, bin_cos, bindistance, bin_dist, source_names[0], 8, makelabel, temp_legend_colors, temp_legend_names)
        
        plt.subplot(3, 2, 2)
        plotFunctions.scatter_maker('dist_0', 'theta_rec_0', data_dict, bin_cos, bindistance, source_names[i], colors[i], 8, makelabel, temp_legend_colors, temp_legend_names)
        plt.subplot(3,2,4)
        plotFunctions.multi_hist('dist_0', 'theta_rec_0', data_dict, bin_cos, bindistance, bin_dist, source_names[i], 8, makelabel, temp_legend_colors, temp_legend_names)

        plt.subplot(3,1,3)
        plotFunctions.diff_hist('dist_0', 'theta_rec_0', source_names[0], source_names[i], source_names, bin_dist, bin_cos, source_names, data_dict, 8, makelabel, custom_lines_color, legend_names)
        plt.savefig('test_plots/MultiHist_{0}_{1}.pdf'.format(source_names[0],source_names[i]), dpi=300)
        plt.clf()
print("Done!") 
print('\n')

IceVolume = 8.4823 * 10

#Writing antenna event and effective volume information to a txt file
print("Writing Event and Effective Volume Data to a TXT File!")
with open('test_plots/All_Event_And_Effective_Volume_Data.txt', 'w') as txtFile:
        txtFile.write('Event and Effective Volume information for each antenna:')
        txtFile.write('\n')
        for i in range(len(source_names)):
                plotFunctions.antenna_data_txt_writer(txtFile, data_dict, source_names, i, IceVolume)
print("Done!")
print('\n')

#Writing the exact same data to a csv file
print("Writing Event and Effective Volume Data to a CSV File!")
with open('test_plots/All_Event_And_Effective_Volume_Data_Temp.txt', 'w') as txtFile:
        txtFile.write('Antenna, Total Events, Triggered, Usable, Weighted, Effective Volume')
        txtFile.write('\n')
        for i in range(len(source_names)):
                plotFunctions.antenna_data_txt_writer_temp(txtFile, data_dict, source_names, i, IceVolume)
csvData = pd.read_csv('test_plots/All_Event_And_Effective_Volume_Data_Temp.txt')
csvData.to_csv('test_plots/All_Event_And_Effective_Volume_Data.csv', index = None)
os.remove('test_plots/All_Event_And_Effective_Volume_Data_Temp.txt')
print("Done!")
print('\n')

for i in range(len(source_names)):
        print('#'*50)
        print('\033[1;37m{0}\033[0;0m'.format(source_names[i]))
        print('#'*50)
        print('Total Events: \033[1;31m{0}\033[0;0m'.format(data_dict[source_names[i]]['Total_Events']))
        print('Triggered: \033[1;31m{0}\033[0;0m'.format(len(data_dict[source_names[i]]['trigg'])))
        print('Usable: \033[1;31m{0}\033[0;0m'.format(len(data_dict[source_names[i]]['weight'])))
        print('Weighted: \033[1;31m{0}\033[0;0m'.format(np.sum(data_dict[source_names[i]]['weight'])))
        print('Effective Volume: \033[1;31m{0}\033[0;0m'.format(IceVolume * 4.0 * np.pi * (
                np.sum(data_dict[source_names[i]]['weight'])/data_dict[source_names[i]]['Total_Events'])))
        print('#'*50)
        print('\n')


#We need the following data for each antenna: theta_rec_0, nnu_theta, weights

if len(source_names) == 2:
        #Collecting the data we need for the first antenna:
        evolved_theta_rec = data_dict[source_names[0]]['theta_rec_0']
        evolved_nnu_theta = data_dict[source_names[0]]['nnu_theta']
        evolved_weights = data_dict[source_names[0]]['weight']

        #Collecting the data we need for the second antenna:
        bicone_theta_rec = data_dict[source_names[1]]['theta_rec_0']
        bicone_nnu_theta = data_dict[source_names[1]]['nnu_theta']
        bicone_weight = data_dict[source_names[1]]['weight']

        #Plotting theta_rec:
        ang_forHist = np.cos(evolved_theta_rec)
        ang_forHist2 = np.cos(bicone_theta_rec)

        theta_recHist1 = -np.cos(evolved_theta_rec)
        theta_recHist2 = -np.cos(bicone_theta_rec)

        nuWeights = evolved_weight
        nuWeights2 = bicone_weights

        numBins=25

        fig = plt.figure(figsize = (10, 10))

        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()


        ax1.hist(theta_recHist1,weights=nuWeights, alpha=1.0, bins=numBins,histtype='step', stacked=True,density=False, fill=False,label=source_names[0], linewidth = 2)
        ax1.hist(theta_recHist2,weights=nuWeights2, alpha=1.0, bins=numBins,histtype='step', stacked=True,density=False, fill=False,label=source_names[1], linewidth = 2)
        #second_x = np.linspace(0, np.pi, 360)
        second_x = np.linspace(180, 0, 180) ## Flipped to be originating direction instead of propagating
        second_y = [100]*180
        ax2.hist(second_x, second_y, label = '_nolegend_', color = 'w', alpha = 0)
        ax1.set_xlabel("$\cos(\\theta_{𝜈})$", fontsize=32, labelpad = 10)
        ax1.set_ylabel("Number of neutrinos / bin", fontsize=32, labelpad = 5)
        ax2.set_xlabel("$\\theta_{𝜈}$", fontsize=32, labelpad = 10)
        #values,binz,patches=axes[0].hist(ang_forHist,weights=nuWeights, range=(-1, 0.5),alpha=1.0, bins=numBins,histtype='step', stacked=True,density=False, fill=False,label="New Bicone");
        #values2,bins2,patches2=axes[0].hist(ang_forHist2,weights=nuWeights2, range=(-1, 0.5),alpha=1.0, bins=binz,histtype='step', stacked=True,density=False, fill=False,label="Old Bicone");

        '''From stack overflow--trying to set scientific notation to use ^3 instead of ^4'''
        class OOMFormatter(matplotlib.ticker.ScalarFormatter):
                def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
                        self.oom = order
                        self.fformat = fformat
                matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
                def _set_order_of_magnitude(self):
                        self.orderOfMagnitude = self.oom
                def _set_format(self, vmin=None, vmax=None):
                        self.format = self.fformat
                if self._useMathText:
                        self.format = r'$\mathdefault{%s}$' % self.format

        #plt.yscale("log")
        #ticks = ax.get_yticks()/1000
        #fig.axes.Axes.set_yticklabels(ticks)
        #plt.xticks(size = 28, ticks = np.arange(-1.0, 1.2, 0.4))
        plt.yticks(size = 26)
        plt.tick_params(axis = 'both', which = 'major', pad = 10)
        plt.margins(0.1)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,4))
        ax1.yaxis.set_major_formatter(OOMFormatter(3, "%1.1f"))
        ax2.yaxis.set_major_formatter(OOMFormatter(3, "%1.1f"))
        ax1.yaxis.offsetText.set_fontsize(16)
        ax1.tick_params(axis = 'both', labelsize = 26)
        ax2.tick_params(labelsize = 26)
        ax1.get_yaxis().get_offset_text().set_position((-0.08, 0.8))
        ax2.yaxis.offsetText.set_fontsize(26)
        #ax2.cla()
        ax1.set_ylim(1E0, 2.2*10**3)
        ax1.set_xlim(-1.2, 1.2)
        ax2.set_xlim( -0.2*90, 180 + 0.2*90)
        my_ticks = [0, 45, 90, 135, 180]
        ax2.set_xticks(my_ticks)
        ax2.set_xticklabels(my_ticks[::-1])
        ax1.grid(linestyle='--', linewidth=1.4)
        #plt.title("Angular distribution of Detected Events", fontsize=26)
        ax1.legend(ncol=6, loc=('upper center'), prop={'size': 25} )
        #plt.suptitle("Angular reconstrucion of simulated events with AraSim", fontsize=22)
        #fig.tight_layout(rect=[0, 0.09, 1, 0.95])
        plt.savefig("test_plots/NuAnglesnew.png", dpi=100, bbox_inches = 'tight')
        plt.clf()

        #Plotting nnu_theta:

        numBins=25#int(tree.GetEntries()/10)

        ang_forHist = evolved_nnu_theta
        ang_forHist2 = bicone_nnu_theta

        nnuHist1 = np.cos(evolved_nnu_theta)
        nnuHist2 = np.cos(bicone_nnu_theta)

        nuWeights = evolved_weight
        nuWeights2 = bicone_weights

        fig = plt.figure(figsize = (10,10))

        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()


        ax1.hist(nnuHist1, weights=nuWeights, bins=numBins, density=False, alpha=1.0,histtype='step',label=source_names[0], linewidth = 2)
        ax1.hist(nnuHist2, weights=nuWeights2, bins=numBins, density=False, alpha=1.0,histtype='step',label=source_names[1], linewidth = 2)
        second_x = np.linspace(180, 0, 180) # Flipped to be propagating instead of originating
        second_y = [100]*180
        ax2.hist(second_x, second_y, label = '_nolegend_', color = 'w', alpha = 0)
        ax1.set_xlabel("$\cos(\\theta_{RF})$", fontsize=32, labelpad = 10)
        ax1.set_ylabel("Number of RF signals / bin", fontsize=32, labelpad = 5)
        ax2.set_xlabel("$\\theta_{RF}$", fontsize = 32, labelpad = 10)
        #bincenters = 0.5*(bins[1:]+bins[:-1])
        #menStd     = np.sqrt(values)
        #menStd2    = np.sqrt(values2)
        width      = 0.1
        #axes[0].bar(bincenters, values,ecolor='C0', edgecolor='C0', yerr=menStd,label="New Bicone", fill=False,width=width)
        #axes[0].bar(bincenters,values2,ecolor='C1',width=width, yerr=menStd2, fill=False,edgecolor='C1',label="Ara Bicone")
        #plt.show()
        #bin_to_integrate = int(len(bins)-125/np.diff(bins)[0])

        #area = sum(np.diff(bins[:bin_to_integrate+1])*values[:bin_to_integrate])
        '''
        axes[0].set_xlabel("$\\theta_{RF}$ [rads]", fontsize=12)
        axes[0].set_ylabel("counts", fontsize=12)
        #axes[0].set_yscale("log")
        #axes[0].set_xlim(-1, 10)
        axes[0].grid(linestyle='--', linewidth=0.8)
        print(np.cos(90))
        values3,binz3,patches3=axes[1].hist(np.cos((ang_forHist)), bins=numBins, density=False,histtype='step',alpha=1.0)
        values4,bins4,patches4=axes[1].hist(np.cos((ang_forHist2)), bins=binz3, density=False,histtype='step',alpha=1.0)
        '''

        '''From stack overflow--trying to set scientific notation to use ^3 instead of ^4'''
        class OOMFormatter(matplotlib.ticker.ScalarFormatter):
                def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
                        self.oom = order
                        self.fformat = fformat
                matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
                def _set_order_of_magnitude(self):
                        self.orderOfMagnitude = self.oom
                def _set_format(self, vmin=None, vmax=None):
                        self.format = self.fformat
                if self._useMathText:
                        self.format = r'$\mathdefault{%s}$' % self.format

        plt.xticks(size = 26, ticks = np.arange(-1.0, 1.2, 0.4))
        plt.yticks(size = 26)
        plt.tick_params(axis = 'both', which = 'major', pad = 5)
        plt.margins(0)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,4))
        ax1.yaxis.set_major_formatter(OOMFormatter(3, "%1.1f"))
        ax2.yaxis.set_major_formatter(OOMFormatter(3, "%1.1f"))
        ax1.yaxis.offsetText.set_fontsize(16)
        ax1.tick_params(axis = 'both', labelsize = 26)
        ax2.tick_params(labelsize = 26)
        ax1.get_yaxis().get_offset_text().set_position((-0.08, 0.8))
        ax2.yaxis.offsetText.set_fontsize(32)
        ax1.set_ylim(1E0, 2.2*10**3)
        ax1.set_xlim(-1.2, 1.2)
        ax2.set_xlim( -0.2*90, 180 + 0.2*90)
        my_ticks = [0, 45, 90, 135, 180]
        ax2.set_xticks(my_ticks)
        ax2.set_xticklabels(my_ticks[::-1])
        ax1.grid(linestyle='--', linewidth=1.4)
        #plt.title("Angular distribution of Detected Events", fontsize=26)
        ax1.legend(ncol=6, loc=('upper center'), prop={'size': 25} )
        #plt.suptitle("Angular reconstrucion of simulated events with AraSim", fontsize=22)
        #fig.tight_layout(rect=[0, 0.09, 1, 0.95])
        #plt.savefig("test_plots/NuAnglesnew.png", dpi=100)


        #bincenters3 = 0.5*(bins3[1:]+bins3[:-1])
        '''
        menStd3 = np.sqrt(values3)
        menStd4 = np.sqrt(values4)
        #axes[1].bar(bincenters3, values3, width=.03, ecolor='C0',fill=False,edgecolor='C0',yerr=menStd3)
        #axes[1].bar(bincenters3,values4,width=.03,ecolor='C1', fill=False,edgecolor='C1',yerr=menStd4)

        # axes[1].set_title("Histogram of $\cos(\\theta_{rec})$", fontsize=12)
        axes[1].set_xlabel("$\cos(\\theta_{RF})$", fontsize=12)
        #axes[1].set_yscale("log")
        #axes[1].set_ylim(5E-2, 1E1)
        axes[1].grid(linestyle='--', linewidth=0.8)
        print(np.cos(np.pi))
        '''
        #fig.suptitle("RF Arrival Angles (Not Normalized)", fontsize=15)
        #fig.tight_layout(rect=[0, 0.09, 1, 0.95])
        fig.savefig("test_plots/ArrivalAngleRFnew.png", dpi = 100, bbox_inches = 'tight')
        plt.clf()

stop = timeit.default_timer()
print('Time: \033[1;31m{0}\033[0;0m'.format(stop - start))
print('\n')
exit()

'''
I think this is the equation AraSim uses:
Veff_test = IceVolume * 4. * PI * Total_Weight / (double)(settings1->NNU);
The IceVolume would be 4/3pi*R^3, and I think R is 3000 m, and then NNU for each root file should be 30000 (so 3*10^6 for each individual)
'''
