#Symulacja - raport 1

path = r'C:\Users\Alicja Kocieniewska\Documents\Uczelnia\ZMS'

import csv
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.stats import kstest
import pandas as pd

damages = {0 : 3437, 
                1 : 522, 
                2 : 40, 
                3 : 2, 
                4 : 0, 
                5 : 0}

plt.bar(list(damages.keys()), 
        list(damages.values()))
plt.title("Wykres słupkowy przedstawiający liczbę szkód o określonej wielkości")
plt.show()

#plt.savefig('liczebnosc_szkody.png')

# calculating average damage:
damages_avg = (sum([x * y for x, y in damages.items()]) / 
                        sum(damages.values()))

# checking if Poisson:
poisson_test = [sc.stats.poisson.pmf(i, damages_avg) * 
                sum(damages.values()) for i in range(len(damages))]

plt.bar(list(damages.keys()), poisson_test, color = "orange")
plt.show()

# testing it:
test1 = sc.stats.chisquare(list(damages.values()), f_exp = poisson_test)
if test1[1] > 0.05:
    print ("brak podstaw do odrzucenia hipotezy zerowej")
else:
    print("należy odrzucic hipoteze zerowa")

   
damage_size = []
with open(path+'\\szkody.txt','r') as csvfile:
    reader = csv.reader (csvfile, delimiter=";")
    for row in reader:
        damage_size.append(int(row[1]))

plt.hist(damage_size, bins=50)
plt.show()
sns.distplot(damage_size, kde=True, color = 'purple', hist=False, 
             kde_kws={'shade': True,'linewidth': 2})
plt.title('Funkcja gęstości przybliżona z użyciem jądra normalnego')
plt.show()
#plt.savefig('gestosc_szkody.png')

print ("Srednia wielkosc szkod:", round(sc.mean(damage_size))) 

# checking if log_normal distribution:
damage_size_ln = sc.log(damage_size)

plt.hist(damage_size_ln, bins=50, color='lightgreen')
plt.title('Histogram rozkładu')
plt.show()
#plt.savefig('hist_damage_size_ln.png')

sns.distplot(damage_size_ln, kde=True, 
             bins=30, color = 'purple',
             hist=True,
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.title('Histogram i funkcja gęstości \n przybliżona z użyciem jądra normalnego')
plt.show() 
#plt.savefig('hist_density_damage_size_ln.png')

sns.distplot(damage_size_ln, kde=True, color = 'blue', hist=False, 
             kde_kws={'shade': True,'linewidth': 2})
plt.title('Funkcja gęstości przybliżona z użyciem jądra normalnego')
plt.show()
#plt.savefig('density_damage_size_ln.png')

sns.distplot(damage_size_ln, kde=False, bins=20,
             fit=sc.stats.norm, color = 'blue', 
             hist_kws={'edgecolor':'gray'}, fit_kws={"color":"red", 'linewidth': 3})
plt.title('Porównanie z rozkładem normalnym')
plt.show()   
#plt.savefig('damage_size_ln_fitting.png')

#TEST: 

test2 = kstest(damage_size_ln, sc.stats.norm.cdf, 
               args = (sc.mean(damage_size_ln), sc.std(damage_size_ln)))
if test2[1] > 0.05:
    print ("p-value wyniosło:", round(test2[1], 4), 
           "- brak więc podstaw do odrzucenia hipotezy " +
           "o log-normalności rozkładu zmiennej")
else:
    print ("należy odrzucic hipotezę zerowa")
    
    
# additional parameters:
damage_size_ln_avg = sc.mean(damage_size_ln)
damage_size_ln_std = sc.std(damage_size_ln)

#simulation model: 

seed = sc.random.seed(5) 

def model (client_num, damage_size_ln_avg, 
           damage_size_ln, damage_size_ln_std, horizon, 
           surplus, contribution, seed):
    contract_dates = [sc.random.randint(0, 364) for i in range(client_num)]
    calendar_in = [0]*365
    for contractDate in contract_dates:
        calendar_in[contractDate] += 1

    damages_for_client = sc.random.poisson(damages_avg, 
                                             client_num)
    
    calendar_out = [0]*(365*horizon) 
    for k in range(client_num):
        for s in range(damages_for_client[k]):
            payment = contract_dates[k] + sc.random.randint(0, 364)
            calendar_out[payment] += 1
    
    
    for day in range(365*horizon):
        if day <= 364:
            surplus += calendar_in[day] * contribution
        withdrawals_num = calendar_out[day]
        compensations = 0 
        if withdrawals_num > 0:
            compensations = sum(sc.exp(sc.random.normal(damage_size_ln_avg, 
                                                        damage_size_ln_std, 
                                                        withdrawals_num)))
        if surplus < compensations:
            return surplus - compensations
        else:
            pass
        surplus -= compensations
    return surplus


#make it multiple: 
    
def func_call(surplus, contribution, n, 
              client_num, damages_avg, 
              damage_size_ln_avg, damage_size_ln_std, horizon):
    result = []
    bankruptcy = 0
    positive_result = []
    for seed in range(n):
        result.append(model(client_num, damage_size_ln_avg, 
                             damage_size_ln, damage_size_ln_std, horizon, 
                             surplus, contribution, seed))
        if result[seed] < 0:
            bankruptcy += 1
        if result[seed] > 0:
            positive_result.append(result[seed])
    result_avg = sc.mean(positive_result)
    bankruptcy_prob = bankruptcy / n
    return [bankruptcy, bankruptcy_prob, result_avg]





#SIMULATION
avg_result = [] 
contribution_size = []
bankruptcy_prob = []
ruins_num = [] 
surplus_level = []
simulation_num = []

client_num = 100
horizon = 2

for n in [25, 150, 1000, 3000]:
    for surplus in [10000]:
        for contribution in range(500,1500,100):
            sim_output = func_call(surplus, contribution, n, 
                                   client_num, damages_avg, 
                                   damage_size_ln_avg, damage_size_ln_std, 
                                   horizon)
            simulation_num.append(n)
            surplus_level.append(surplus)
            contribution_size.append(contribution)
            ruins_num.append(sim_output[0])
            bankruptcy_prob.append(sim_output[1])
            avg_result.append(sim_output[2])
            print("Nadwyzka: ", surplus, "Skladka: ", contribution, 
            "Liczba ruin: ", sim_output[0], "Sredni wynik: ",
            round(sim_output[2]), "bankruptcy_prob: ", sim_output[1])
            

























