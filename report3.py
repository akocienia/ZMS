import numpy as np 
import random 
from math import cos, sin, sqrt

# MODEL FUNCTION

def model(horizon, number_of_alpacas, number_of_water_spots, setup):

    # number of 15min periods 
    # during the day only - alpacas don't drink at night - they sleep :)
    
    checkpoints = int((horizon * 24 * 4) / 2)
    
    # starting location of all our alpacas (they are on the circle)
    alpacas_loc_start = [[random.random(), random.randint(0, 360)] \
                          for alpaca in range(number_of_alpacas)]
    
    x_start = [loc[0] * cos(loc[1]) for loc in alpacas_loc_start]
    y_start = [loc[0] * sin(loc[1]) for loc in alpacas_loc_start]
    
    alpacas_loc_start = [[x_start[i], y_start[i]] for i in range(len(x_start))]
    
    # we start the simulation with all alpaca power. 
    # all alpacas are full of energy and they don't need any water 

    alpaca_power = 100 * number_of_alpacas
    
    thirsty_alpacas = 0
    
    alpacas_water_need = [0 for alpaca in range(number_of_alpacas)]
    
    # temperature differs between 20'C and 30'C
    
    temperature = random.randint(20, 30)
    
    # we check the situation every 15 minutes
    
    for checkpoint in range(checkpoints):
            
        # alpacas are walking randomly, we check their location every 15 min
        
        alpacas_new_location = [[random.random(), random.randint(0, 360)] \
                                 for alpaca in range(number_of_alpacas)]
        
        x = [loc[0] * cos(loc[1]) for loc in alpacas_new_location]
        y = [loc[0] * sin(loc[1]) for loc in alpacas_new_location]
    
        alpacas_new_location = [[x[i], y[i]] for i in range(number_of_alpacas)]
        
        # and we calculate their distances from the previous locations
                
        distance = [sqrt((x[i] - x_start[i])**2 + (y[i] - y_start[i])**2) \
                     for i in range(number_of_alpacas)]
        
        # every 3 hour we update the temperature 
        
        if checkpoint % 12 == 0:
            temperature = random.randint(20, 30)
            
        # every day we update their alpacoutility 
        
        if checkpoint % 4*12 == 0:
            alpaca_power += 100 * number_of_alpacas
            
            
        # and we update the level of alpacas' thirst
        # it depends on temperature and distance and is based on gamma distribution
            
        water_need_increase = [np.random.gamma(temperature) * 0.01 * distance[alpaca] \
                               for alpaca in range(number_of_alpacas)]
            
        alpacas_water_need = [alpacas_water_need[alpaca] + water_need_increase[alpaca] \
                              for alpaca in range(number_of_alpacas)]
        
        # don't know where to place this variable xD
        
        drinking_time = []
        
        # we check if alpacas need to go to the water spot
        if setup == "lightbulb":
            for i in range(number_of_alpacas):
                    # location of 4 water spots in lightbulb setup 
                    water_spot = [0, -1]
                    # if alpaca is thirsty
                    if alpacas_water_need[i] > 1:
                    # it makes a trip to water spot
                        water_trip = sqrt((water_spot[0] - x[i])**2 + (water_spot[1] - y[i])**2)
                    # and belongs to thirsty alpacas team 
                        thirsty_alpacas += 1
                    # it changes its location to water spot 
                        alpacas_new_location[i] = water_spot
                    # and it's thirst increase (cause it needs to go to new spot)
                        increase_after_trip = np.random.gamma(temperature) * 0.01 * water_trip 
                        alpacas_water_need[i] +=  increase_after_trip
                    # duration of drinking depends of how thirsty the alpaca is
                        drinking_time.append(5 * alpacas_water_need[i])
                        alpacas_water_need[i] = 0
            # if there are more alpacas in the queue, they fight 
            # and loose some alpaca power points 
            if thirsty_alpacas > number_of_water_spots:
                alpaca_power -=  ((10 + 10 * thirsty_alpacas) * thirsty_alpacas)/2
                drinking_time.sort()
                # thirsty alpacas don't have power to fight for better place in queue
                #alpaca_power -= sum(drinking_time[:thirsty_alpacas+1])
                #thirsty_alpacas -= number_of_water_spots
            #thirsty_alpacas = 0
        # do dokończenia kejs z odejmowaniem punktów za czas oczekiwania
        if setup == "hedgehog":
            pass
            
        # that needs to be at the end of the big loop (so that we don't forget)            
        alpacas_loc_start = alpacas_new_location
    
    
    return(alpaca_power)

                
                    
        

            
        
        
        

        
        
    
        
    
    

