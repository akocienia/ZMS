#Report2 

import numpy as np  

def model(horizon, avg_working_time, avg_repair_time, n, m, setup):
    # setup - "L" or "G"
    # horizon in minutes
    horizon = horizon * 24 * 60 
    events = list(np.random.exponential(avg_working_time, n))
    status = ["W"] * n 
    t_start = [0] * n
    t_cum = [0] * n

    # machine number or -1 (not in use)
    tools_loc = [-1] * m
    tools_occupied = [0] * m
    t = min(events)
        
    while t <= horizon:
        for i in range(m):
            if tools_occupied[i] <= t:
                tools_loc[i] = -1
        machine = events.index(t) 

        if status[machine] == "W":
            t_start[machine] = t
            tools = - 1
            for i in range(m):
                if tools_loc[i] == -1:
                    tools = i
                    break
            if tools == -1 :
                status[machine] = "Q"
                events[machine] = min(tools_occupied)
            else:
                status[machine] = "R"
                if setup == "L":
                    transport_time = 2 * (1 + machine)
                elif setup == "G":
                    transport_time =  3
                else:
                    print("Niepoprawny układ! Należy wybrać układ 'L' lub 'G'!")
                    break
                repair_time = np.random.gamma(3, avg_repair_time/3)
                events[machine] += repair_time + transport_time
                tools_loc[tools] = machine
                tools_occupied[tools] += repair_time + 2 * transport_time
                
        elif status[machine] == "Q":
            for i in range(m):
                if tools_loc[i] == -1:
                    tools = i
                    break
            status[machine] = "R"
            if setup == "L":
                transport_time = 2 * (1 + machine)
            elif setup == "G":
                transport_time =  3
            else:
                print("zly uklad - moze byc L lub G!")
                break
            repair_time = np.random.gamma(3, avg_repair_time/3)
            events[machine] += repair_time + transport_time
            tools_loc[tools] = machine
            tools_occupied[tools] += repair_time + 2 * transport_time 
        else:
            status[machine] = "W"
            events[machine] += np.random.exponential(avg_working_time)
            t_cum[machine] += t - t_start[machine]
        
        t = min(events)
        
    return (t_cum)


def run_model (iterations, horizon, avg_working_time, avg_repair_time, n, m, setup):
    avg_t_cum = []
    for i in range (iterations):
        avg_t_cum.append(model( horizon, avg_working_time, avg_repair_time, n, m, setup))
    return list(map(np.mean, np.transpose(avg_t_cum)))

#variables
    
n = 6 # number of machines
avg_working_time = 75 
avg_repair_time = 15
m = 8 #number of tools packages
horizon = 30
iterations = 1000

#Simulation

all_results = [run_model(iterations, horizon, avg_working_time, avg_repair_time, n, m, setup) 
for setup in ["G", "L"] for m in range(1, 9)]

results_mean = [np.mean(i) for i in all_results]

#Which setup is better? Visualisation

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

star_setup = results_mean[0:6]
line_setup = results_mean[8:14]
elements1 = np.arange(len(star_setup))
elements2 = [x + .4 for x in elements1]

fig=plt.figure(figsize=(7, 5)) 
plt.bar(elements2, star_setup, color='orange', edgecolor='white', width=0.45, label="układ gniazdowy")
plt.bar(elements1, line_setup, color='dodgerblue', edgecolor='white', width=0.45, label="układ liniowy")
plt.xticks([r + 0.25 for r in range(len(elements1))], ['1', '2', '3', '4', '5', '6'])
plt.title('Porównanie przeciętnego czasu dla obu układów',fontsize=18)
plt.xlabel('liczba paczek z narzędziami', )
plt.ylabel("przeciętny czas oczekiwania")
plt.legend(loc = "best", edgecolor = "black", shadow=True)
plt.show()

elements = range(1, 7)

plt.plot(elements, results_mean[0:6], color="orange", marker='o', alpha=0.8, label="Układ gniazdowy")
plt.plot(elements, results_mean[8:14],  marker='o', color="dodgerblue", alpha=0.8, label = "Układ liniowy")
plt.title('Przeciętny czas oczekiwania dla paczek z pierwszą włącznie',fontsize=14)
plt.legend(loc='best', edgecolor = "black", shadow=True)
plt.show()

#starting from 2 packages to visualize better: 

elements_from_2 = range(2, 7)

fig=plt.figure(figsize=(9.5, 9.5)) 
plt.subplot(221)
plt.plot(elements_from_2, results_mean[1:6], color="orange", marker='o', alpha=0.8)
plt.xticks(np.arange(2, 7, 1))
plt.title('Układ gniazdowy', fontsize=12, color='grey', loc='left', style='italic')
plt.ylabel('przeciętny czas oczekiwania')
plt.xlabel('liczba paczek z narzędziami')
plt.subplot(222)
plt.plot(elements_from_2, results_mean[9:14],  marker='o', color="dodgerblue", alpha=0.8)
plt.xticks(np.arange(2, 7, 1))
plt.xlabel('liczba paczek z narzędziami')
plt.title('Układ liniowy', fontsize=12, color='grey', loc='left', style='italic')    
# Add a title:
plt.suptitle('Przeciętny czas oczekiwania dla paczek z pierwszą włącznie',fontsize=14, y=0.95)
plt.show()

