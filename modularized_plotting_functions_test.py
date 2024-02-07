#Jack Tillman
#This python script contains the functions used in the "Modularized_PlotMaker_Test.py" file

#Importing the necessary python and ROOT libraries
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import numpy as np
import argparse
import warnings
import ROOT
from ROOT import gInterpreter, gSystem
from ROOT import TChain
#Defining a function to create the lists used in the loop based off of the parsed data:

def parsed_data_list(g):
        #Making a dictionary of the parsed arguments
        source_dict = g.__dict__
        #Deleting empty arguments from dictionary
        source_dict = {k:v for k, v in source_dict.items() if v != None}
        print('#'*50)
        print(source_dict)
        print('#'*50)
        print('\n')

        source_names = list(source_dict.keys())
        for i in range(len(source_names)):
                source_names[i] = source_dict[source_names[i]][0].split('.')[-4]
        return source_dict, source_names

#Defining a function for the data analysis & extraction loop:

def data_analysis(source_dict, source_names):
        energy = np.power(10,18)
        earth_depth = 6359632.4
        core_x = 10000.0
        core_y = 10000.0

        data_dict = {}
        for i in range(len(source_dict.keys())):
        
                #General info for each simulation set
                print('#'*50)
        
                #setting trees
                var_dict = {}
                #list of all variable names
                var = ['trigg', 'weight', 'posnu_x', 'posnu_y', 'posnu_z',
                'rec_ang_0', 'theta_rec_0', 'reflect_ang_0',
                'dist_0', 'arrival_time_0', 'reflection_0', 
                'l_att_0', 'view_ang_0', 'launch_ang_0',
                'rec_ang_1', 'theta_rec_1', 'reflect_ang_1',
                'dist_1', 'arrival_time_1', 'reflection_1', 
                'l_att_1', 'view_ang_1', 'launch_ang_1',
                'current', 'flavor', 'elast',
                'nnu_theta', 'nnu_phi', 'ShowerEnergy',
                'depth', 'distance']
        
                #loop for making dictionary of variables and empty list
                for x in var:
                        var_dict['{0}'.format(x)] = []

                SimTree = [] #sets SimTree and makes empty list
                SimTree = TChain("AraTree2") #Which tree I will be chaining
                for line in list(source_dict.values())[i]: #for every file name in my list
                        SimTree.AddFile(line)
                reportPtr = ROOT.Report()#report pointer
                eventPtr = ROOT.Event()#event pointer
                #detectorPtr = ROOT.Detector()
                #can also add more pointers if needed
                SimTree.SetBranchAddress("report", ROOT.AddressOf(reportPtr))
                SimTree.SetBranchAddress("event", ROOT.AddressOf(eventPtr))
        
                #basic info of data
                totalEvents = SimTree.GetEntries()
        
                print('\033[1;37m{0}\033[0;0m'.format(source_names[i]))
                print('Total Events: {0}'.format(totalEvents))
                print('#'*50)
                var_dict['Total_Events'] = []
                var_dict['Total_Events'] = totalEvents
                var_dict['Total_Weights'] = []
        
        ##Beaks here##
                #Now we loop over all the events 
                for j in range(totalEvents):
                        SimTree.GetEntry(j)
                        var_dict['Total_Weights'].append(eventPtr.Nu_Interaction[0].weight)

                        #Selecting only triggered events and a weight between 0 and 1
                        if (reportPtr.stations[0].Global_Pass > 0) and (eventPtr.Nu_Interaction[0].weight >= 0 and eventPtr.Nu_Interaction[0].weight <= 1):
                                trigg = j
                                var_dict['trigg'].append(j)
                        
                                #If value is seen in both antennas (Top Vpol and Bot Vpol) then we take an average of two
                                try:                                                                 
                                        #interaction position in ice
                                        posnu_x = eventPtr.Nu_Interaction[0].posnu.GetX()
                                        posnu_y = eventPtr.Nu_Interaction[0].posnu.GetY()
                                        posnu_z = eventPtr.Nu_Interaction[0].posnu.GetZ()
                                
                                        #Getting angle of received signal in antenna
                                        #Direct solutions
                                        rec_ang_0 = ((reportPtr.stations[0].strings[1].antennas[0].rec_ang[0] + 
                                                reportPtr.stations[0].strings[1].antennas[2].rec_ang[0])/2.0)
                                        reflect_ang_0 = ((reportPtr.stations[0].strings[1].antennas[0].reflect_ang[0] +
                                                reportPtr.stations[0].strings[1].antennas[2].reflect_ang[0])/2.0)
                                        theta_rec_0 = ((reportPtr.stations[0].strings[1].antennas[0].theta_rec[0] +
                                               reportPtr.stations[0].strings[1].antennas[2].theta_rec[0])/2.0)
                                
                                        dist_0 = reportPtr.stations[0].strings[1].antennas[0].Dist[0]
                                        arrival_time_0 = reportPtr.stations[0].strings[1].antennas[0].arrival_time[0] 
                                        reflection_0 = reportPtr.stations[0].strings[1].antennas[0].reflection[0]
                                        l_att_0 = reportPtr.stations[0].strings[1].antennas[0].L_att[0]
                                        view_ang_0 = reportPtr.stations[0].strings[1].antennas[0].view_ang[0]
                                        launch_ang_0 = reportPtr.stations[0].strings[1].antennas[0].launch_ang[0]

                                        #Refracted/Reflected solutions
                                        rec_ang_1 = ((reportPtr.stations[0].strings[1].antennas[0].rec_ang[1] +
                                             reportPtr.stations[0].strings[1].antennas[2].rec_ang[1])/2.0)
                                        reflect_ang_1 = ((reportPtr.stations[0].strings[1].antennas[0].reflect_ang[1] +
                                                 reportPtr.stations[0].strings[1].antennas[2].reflect_ang[1])/2.0)
                                        theta_rec_1 = ((reportPtr.stations[0].strings[1].antennas[0].theta_rec[1] +
                                               reportPtr.stations[0].strings[1].antennas[2].theta_rec[1])/2.0)
                                
                                        dist_1 = reportPtr.stations[0].strings[1].antennas[0].Dist[1]
                                        arrival_time_1 = reportPtr.stations[0].strings[1].antennas[0].arrival_time[1] 
                                        reflection_1 = reportPtr.stations[0].strings[1].antennas[0].reflection[1]
                                        l_att_1 = reportPtr.stations[0].strings[1].antennas[0].L_att[1]
                                        view_ang_1 = reportPtr.stations[0].strings[1].antennas[0].view_ang[1]
                                        launch_ang_1 = reportPtr.stations[0].strings[1].antennas[0].launch_ang[1]
                                        #incoming neutrino info
                                        nnu_theta = eventPtr.Nu_Interaction[0].nnu.Theta()
                                        nnu_phi = eventPtr.Nu_Interaction[0].nnu.Phi()
                                        current = eventPtr.Nu_Interaction[0].currentint
                                        flavor = eventPtr.nuflavorint
                                        elast = eventPtr.Nu_Interaction[0].elast_y
                                        #weight
                                        weight = eventPtr.Nu_Interaction[0].weight                
                                        if current == 1 and flavor == 1:
                                                ShowerEnergy = energy                                        
                                        else:
                                                ShowerEnergy = energy * elast
                                
                                        depth = posnu_z - earth_depth
                                        distance =  ((posnu_x - core_x)**2 + (posnu_y - core_y)**2 )**(0.5)
                                        all_var = [trigg, weight, posnu_x, posnu_y, posnu_z,
                                        rec_ang_0, theta_rec_0, reflect_ang_0,
                                        dist_0, arrival_time_0, reflection_0, 
                                        l_att_0, view_ang_0, launch_ang_0,
                                        rec_ang_1, theta_rec_1, reflect_ang_1,
                                        dist_1, arrival_time_1, reflection_1, 
                                        l_att_1, view_ang_1, launch_ang_1,
                                        current, flavor, elast,
                                        nnu_theta, nnu_phi, ShowerEnergy,
                                        depth, distance]

                                        for k in range(1,len(all_var)):
                                                var_dict['{0}'.format(var[k])].append(all_var[k])
                                        print(j)

                                except IndexError:
                                
                                        #Both antennas didn't see a signal, so we try with index 0 (Bot Vpol)
                                        try: 
                                        
                                                #interaction position in ice
                                                posnu_x = eventPtr.Nu_Interaction[0].posnu.GetX()
                                                posnu_y = eventPtr.Nu_Interaction[0].posnu.GetY()
                                                posnu_z = eventPtr.Nu_Interaction[0].posnu.GetZ()
                                        
                                                #angles seen by antenna
                                                rec_ang_0 = reportPtr.stations[0].strings[1].antennas[0].rec_ang[0]
                                                theta_rec_0 = reportPtr.stations[0].strings[1].antennas[0].theta_rec[0]
                                                reflect_ang_0 = reportPtr.stations[0].strings[1].antennas[0].reflect_ang[0]
                                        
                                                dist_0 = reportPtr.stations[0].strings[1].antennas[0].Dist[0]
                                                arrival_time_0 = reportPtr.stations[0].strings[1].antennas[0].arrival_time[0] 
                                                reflection_0 = reportPtr.stations[0].strings[1].antennas[0].reflection[0]
                                                l_att_0 = reportPtr.stations[0].strings[1].antennas[0].L_att[0]
                                        
                                                view_ang_0 = reportPtr.stations[0].strings[1].antennas[0].view_ang[0]
                                                launch_ang_0 = reportPtr.stations[0].strings[1].antennas[0].launch_ang[0]
                                       
                                                rec_ang_1 = reportPtr.stations[0].strings[1].antennas[0].rec_ang[1]
                                                theta_rec_1 = reportPtr.stations[0].strings[1].antennas[0].theta_rec[1]
                                                reflect_ang_1 = reportPtr.stations[0].strings[1].antennas[0].reflect_ang[1]
                                        
                                                #other info 
                                        
                                                dist_1 = reportPtr.stations[0].strings[1].antennas[0].Dist[1]
                                                arrival_time_1 = reportPtr.stations[0].strings[1].antennas[0].arrival_time[1] 
                                                reflection_1 = reportPtr.stations[0].strings[1].antennas[0].reflection[1]
                                                l_att_1 = reportPtr.stations[0].strings[1].antennas[0].L_att[1]
                                        
                                                view_ang_1 = reportPtr.stations[0].strings[1].antennas[0].view_ang[1]
                                                launch_ang_1 = reportPtr.stations[0].strings[1].antennas[0].launch_ang[1]       
                                        
                                                #incoming neutrino info
                                                nnu_theta = eventPtr.Nu_Interaction[0].nnu.Theta()
                                                nnu_phi = eventPtr.Nu_Interaction[0].nnu.Phi()
                                        
                                                current = eventPtr.Nu_Interaction[0].currentint
                                                flavor = eventPtr.nuflavorint
                                                elast = eventPtr.Nu_Interaction[0].elast_y
                                        
                                                #weight
                                                weight = eventPtr.Nu_Interaction[0].weight
                                                
                                                if current == 1 and flavor == 1:
                                                        ShowerEnergy = energy                                        
                                                else:
                                                        ShowerEnergy = energy * elast
                                                
                                                depth = posnu_z - earth_depth
                                                distance = ((posnu_x - core_x)**2 + (posnu_y - core_y)**2 )**(0.5)
                                                                                        
                                                all_var = [trigg, weight, posnu_x, posnu_y, posnu_z,
                                                        rec_ang_0, theta_rec_0, reflect_ang_0,
                                                        dist_0, arrival_time_0, reflection_0, 
                                                        l_att_0, view_ang_0, launch_ang_0,
                                                        rec_ang_1, theta_rec_1, reflect_ang_1,
                                                        dist_1, arrival_time_1, reflection_1, 
                                                        l_att_1, view_ang_1, launch_ang_1,
                                                        current, flavor, elast,
                                                        nnu_theta, nnu_phi, ShowerEnergy,
                                                        depth, distance]
                                                
                                                for k in range(1,len(all_var)):
                                                        var_dict['{0}'.format(var[k])].append(all_var[k])
                                                
                                                print(str(j)+" only has Bot Vpol signal")

                                        except IndexError:
                                                try: #Have this here because not always that both antenna see
                                                
                                                        #interaction position in ice
                                                        posnu_x = eventPtr.Nu_Interaction[0].posnu.GetX()
                                                        posnu_y = eventPtr.Nu_Interaction[0].posnu.GetY()
                                                        posnu_z = eventPtr.Nu_Interaction[0].posnu.GetZ()
                                                
                                                        #angles seen by antenna
                                                        rec_ang_0 = reportPtr.stations[0].strings[1].antennas[2].rec_ang[0]
                                                        theta_rec_0 = reportPtr.stations[0].strings[1].antennas[2].theta_rec[0]
                                                        reflect_ang_0 = reportPtr.stations[0].strings[1].antennas[2].reflect_ang[0]
                                                
                                                        rec_ang_1 = reportPtr.stations[0].strings[1].antennas[2].rec_ang[1]
                                                        theta_rec_1 = reportPtr.stations[0].strings[1].antennas[2].theta_rec[1]
                                                        reflect_ang_1 = reportPtr.stations[0].strings[1].antennas[2].reflect_ang[1]
                                                
                                                        #other info 
                                                        dist_0 = reportPtr.stations[0].strings[1].antennas[2].Dist[0]
                                                        arrival_time_0 = reportPtr.stations[0].strings[1].antennas[2].arrival_time[0] 
                                                        reflection_0 = reportPtr.stations[0].strings[1].antennas[2].reflection[0]
                                                        l_att_0 = reportPtr.stations[0].strings[1].antennas[2].L_att[0]
                                                
                                                        view_ang_0 = reportPtr.stations[0].strings[1].antennas[2].view_ang[0]
                                                        launch_ang_0 = reportPtr.stations[0].strings[1].antennas[2].launch_ang[0]
                                                
                                                        dist_1 = reportPtr.stations[0].strings[1].antennas[2].Dist[1]
                                                        arrival_time_1 = reportPtr.stations[0].strings[1].antennas[2].arrival_time[1] 
                                                        reflection_1 = reportPtr.stations[0].strings[1].antennas[2].reflection[1]
                                                        l_att_1 = reportPtr.stations[0].strings[1].antennas[2].L_att[1]
                                                
                                                        view_ang_1 = reportPtr.stations[0].strings[1].antennas[2].view_ang[1]
                                                        launch_ang_1 = reportPtr.stations[0].strings[1].antennas[2].launch_ang[1]       
                                                
                                                        #incoming neutrino info
                                                        nnu_theta = eventPtr.Nu_Interaction[0].nnu.Theta()
                                                        nnu_phi = eventPtr.Nu_Interaction[0].nnu.Phi()
                                                
                                                        current = eventPtr.Nu_Interaction[0].currentint
                                                        flavor = eventPtr.nuflavorint
                                                        elast = eventPtr.Nu_Interaction[0].elast_y
                                                
                                                        #weight
                                                        weight = eventPtr.Nu_Interaction[0].weight
                                                
                                                
                                                        if current == 1 and flavor == 1:
                                                                ShowerEnergy = energy                                        
                                                        else:
                                                                ShowerEnergy = energy * elast
                                                        
                                                        depth = posnu_z - earth_depth
                                                        distance = ((posnu_x - core_x)**2 + (posnu_y - core_y)**2 )**(0.5)
                                                
                                                        all_var = [trigg, weight, posnu_x, posnu_y, posnu_z,
                                                                rec_ang_0, theta_rec_0, reflect_ang_0,
                                                                dist_0, arrival_time_0, reflection_0, 
                                                                l_att_0, view_ang_0, launch_ang_0,
                                                                rec_ang_1, theta_rec_1, reflect_ang_1,
                                                                dist_1, arrival_time_1, reflection_1, 
                                                                l_att_1, view_ang_1, launch_ang_1,
                                                                current, flavor, elast,
                                                                nnu_theta, nnu_phi, ShowerEnergy,
                                                                depth, distance]
                                                        
                                                        for k in range(1,len(all_var)):
                                                                var_dict['{0}'.format(var[k])].append(all_var[k])
                                                                
                                                
                                                        print(str(j)+" only has Top Vpol signal")                                                             
                                                except IndexError:
                                                        print("Event "+str(j)+" has no signal in either Top or Bot Vpol")
                                                        continue
                                                
        
                #end of loop                                                    
                data_dict['{0}'.format(source_names[i])] = var_dict
                print("#"*50)
                print('\n')
        return var_dict, data_dict, source_names



#Defining the plotting functions used!

#Histogram Plotting Function
def hist_maker(data_dict, bin_cos, bindistance, hist_var, source, color, fontsize, makelabel, custom_lines_color, legend_names):
        try:    
                if 'ang' in hist_var or 'theta' in hist_var or 'phi' in hist_var:
                        plt.hist(np.cos(data_dict[source]['{0}_0'.format(hist_var)]), 
                                 weights=data_dict[source]['weight'],bins=bin_cos, density=False, 
                                 histtype='step', color=color, ls='-', label=str(source)+' direct')
                        #plt.hist(np.cos(data_dict[source]['{0}_1'.format(hist_var)]), 
                                 #weights=data_dict[source]['weight'], bins=bin_cos, density=False, 
                                 #histtype='step', color=color, ls='--', label=str(source)+' refracted')
                        plt.xlabel("Cos({0})".format(hist_var), fontsize=fontsize)
                        
                else:
                        plt.hist(data_dict[source]['{0}_0'.format(hist_var)], 
                                 weights=data_dict[source]['weight'],bins=bindistance, density=False, 
                                 histtype='step', color=color, ls='-', label=str(source)+' direct')
                        #plt.hist(data_dict[source]['{0}_1'.format(hist_var)], 
                                 #weights=data_dict[source]['weight'], bins=bindistance, density=False, 
                                 #histtype='step', color=color, ls='--', label=str(source)+' refracted')
                        plt.xlabel("{0}".format(hist_var), fontsize=fontsize)
                        
                plt.ylabel("Events", fontsize=fontsize)
                plt.grid(linestyle='--')
                plt.tight_layout()
                if makelabel is 1:
                        legend = plt.legend(custom_lines_color, legend_names, loc='best')
                        plt.gca().add_artist(legend)
  
        except KeyError:
                
                if 'ang' in hist_var or 'theta' in hist_var or 'phi' in hist_var:
                        plt.hist(np.cos(data_dict[source]['{0}'.format(hist_var)]), 
                                 weights=data_dict[source]['weight'], bins=bin_cos, density=False, 
                                 histtype='step', color=color, ls='-', label=str(source))
                        plt.xlabel("Cos({0})".format(hist_var), fontsize=fontsize)
                        
                elif 'weight' in hist_var:
                        plt.hist(data_dict[source]['{0}'.format(hist_var)], 
                                 log=True, density=False, 
                                 histtype='step', color=color, ls='-', label=str(source))
                        plt.xlabel("{0}".format(hist_var), fontsize=fontsize)
                        
                elif 'ShowerEnergy' in hist_var:
                        plt.hist(data_dict[source]['{0}'.format(hist_var)],
                                 density=False, weights=data_dict[source]['weight'],
                                 histtype='step', log=True, 
                                 color=color, ls='-', label=str(source))
                        plt.xlabel("{0}".format(hist_var), fontsize=fontsize)
                        
                elif 'depth' in hist_var or 'distance' in hist_var:
                        plt.hist(data_dict[source]['{0}'.format(hist_var)],
                                 density=False, weights=data_dict[source]['weight'],
                                 histtype='step', 
                                 color=color, ls='-', label=str(source), bins= 40)
                        plt.xlabel("{0}".format(hist_var), fontsize=fontsize)
                        
                else:
                        plt.hist(data_dict[source]['{0}'.format(hist_var)], 
                                 weights=data_dict[source]['weight'],density=False, 
                                 histtype='step', color=color, ls='-', label=str(source))
                        plt.xlabel("{0}".format(hist_var), fontsize=fontsize)
                        
                plt.ylabel("Events", fontsize=fontsize)
                plt.grid(linestyle='--')
                plt.tight_layout()
                if makelabel is 1:
                        legend = plt.legend(custom_lines_color, legend_names, loc='best')
                        plt.gca().add_artist(legend)
                
 
#Scatterplot Plotting Function
def scatter_maker(var1, var2, data_dict, bin_cos, bindistance, source, color, fontsize, makelabel, custom_lines_color, legend_names):
        if 'ang' in var2 or 'theta' in var2 or 'phi' in var2:
                plt.scatter(data_dict[source]['{0}'.format(var1)],
                            np.cos(data_dict[source]['{0}'.format(var2)]), 
                            s=1.0, alpha=0.25, color=color, label=str(source))
                            
                plt.xlabel("{0}".format(var1), fontsize=fontsize)
                plt.ylabel("Cos({0})".format(var2), fontsize=fontsize)
                
        else:
                        
                plt.scatter(data_dict[source]['{0}'.format(var1)], 
                            data_dict[source]['{0}'.format(var2)], 
                            s=1.0, alpha=0.25, color=color, label=str(source))
         
                plt.xlabel("{0}".format(var1), fontsize=fontsize)
                plt.ylabel("{0}".format(var2), fontsize=fontsize)
        
        plt.title("{0}".format(source), fontsize=fontsize)
        plt.grid(linestyle='--')
        if makelabel is True:
                legend = plt.legend(custom_lines_color, legend_names, loc='best')
                plt.gca().add_artist(legend)
        plt.tight_layout()

#Multi-Histogram Plotting Function
def multi_hist(var1, var2, data_dict, bin_cos, bindistance, bin_dist, source, fontsize, makelabel, custom_lines_color, legend_names):
        hist_dict = {}
        hist = []
        hist = plt.hist2d(data_dict[source]['{0}'.format(var1)], 
                          np.cos(data_dict[source]['{0}'.format(var2)]), 
                          bins=(bin_dist,bin_cos), weights=data_dict[source]['weight'])
        hist_dict[source] = hist
       
        plt.colorbar()
        plt.title("{0}".format(source), fontsize=fontsize)
        plt.xlabel("{0}".format(var1), fontsize=fontsize)
        plt.ylabel("{0}".format(var2), fontsize=fontsize)
        if makelabel is 1:
                legend = plt.legend(custom_lines_color, legend_names, loc='best')
                plt.gca().add_artist(legend)
        plt.tight_layout()

#Difference Histogram Plotting Function (Plots a histogram showing the difference
# between 2 data sets)
def diff_hist(var1, var2, source1, source2, source_names, bin_dist, bin_cos, source, data_dict, fontsize, makelabel, custom_lines_color, legend_names):
        hist_dict = {}

        for j in range(len(source_names)):
                hist = []
                hist = plt.hist2d(data_dict[source[j]]['{0}'.format(var1)], 
                          np.cos(data_dict[source[j]]['{0}'.format(var2)]), 
                          bins=(bin_dist,bin_cos), weights=data_dict[source[j]]['weight'])
                hist_dict[source[j]] = hist


        if len(source_names) > 1:
                diff = hist_dict[source2][0] - hist_dict[source1][0]
                plt.pcolormesh(bin_dist, bin_cos, diff.T, cmap='bwr')
                plt.colorbar()
                plt.xlabel("{0}".format(var1), fontsize=fontsize)
                plt.ylabel("{0}".format(var2), fontsize=fontsize)
                plt.title("{0} vs {1}".format(source1, source2), fontsize=fontsize)
                plt.tight_layout()
        else: 
                print("We can't make a 2D histogram showing a difference, if we only have one dataset...")
        if makelabel is 1:
                legend = plt.legend(custom_lines_color, legend_names, loc='best')
                plt.gca().add_artist(legend)

#Antenna Information Writer (Writes the information about each antenna to a single txt file)
def antenna_data_txt_writer(txtFile, data_dict, source_names, i, IceVolume):
        txtFile.write('\n')
        txtFile.write('#'*37)
        txtFile.write('\n')
        txtFile.write('{0}'.format(source_names[i]))
        txtFile.write('\n')
        txtFile.write('#'*37)
        txtFile.write('\n')
        txtFile.write('Total Events: {0}'.format(data_dict[source_names[i]]['Total_Events']))
        txtFile.write('\n')
        txtFile.write('Triggered: {0}'.format(len(data_dict[source_names[i]]['trigg'])))
        txtFile.write('\n')
        txtFile.write('Usable: {0}'.format(len(data_dict[source_names[i]]['weight'])))
        txtFile.write('\n')
        txtFile.write('Weighted: {0}'.format(np.sum(data_dict[source_names[i]]['weight'])))
        txtFile.write('\n')
        txtFile.write('Effective Volume: {0}'.format(IceVolume * 4.0 * np.pi * (
                np.sum(data_dict[source_names[i]]['weight'])/data_dict[source_names[i]]['Total_Events'])))
        txtFile.write('\n')
        txtFile.write('#'*37)
        txtFile.write('\n')

#Creating a function to write a stripped down text file to convert to a csv file:
def antenna_data_txt_writer_temp(txtFile, data_dict, source_names, i, IceVolume):
        txtFile.write('{0}'.format(source_names[i]))
        txtFile.write(',')
        txtFile.write(' {0}'.format(data_dict[source_names[i]]['Total_Events']))
        txtFile.write(',')
        txtFile.write('{0}'.format(len(data_dict[source_names[i]]['trigg'])))
        txtFile.write(',')
        txtFile.write('{0}'.format(len(data_dict[source_names[i]]['weight'])))
        txtFile.write(',')
        txtFile.write('{0}'.format(np.sum(data_dict[source_names[i]]['weight'])))
        txtFile.write(',')
        txtFile.write('{0}'.format(IceVolume * 4.0 * np.pi * (
                np.sum(data_dict[source_names[i]]['weight'])/data_dict[source_names[i]]['Total_Events'])))
        txtFile.write('\n')
