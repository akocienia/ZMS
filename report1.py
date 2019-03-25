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
            
#Plotting 
            
#A: Does simulation number have any impact? 
            
fig=plt.figure(figsize=(9, 9.5)) 
plt.subplot(221)
plt.plot( contribution_size[:10], bankruptcy_prob[:10], color="red", marker='o', alpha=0.4)
plt.title('30 powtórzeń symulacji', fontsize=12, color='grey', loc='left', style='italic')
plt.ylabel('Prawdopodobieństwo bankructwa')
plt.xticks(np.arange(min(contribution_size[:10]), max(contribution_size[:10])+1, 100), rotation = 45)
plt.subplot(222)
plt.plot(contribution_size[10:20], bankruptcy_prob[10:20],  marker='o', color="blue", alpha=0.3)
plt.title('100 powtórzeń symulacji', fontsize=12, color='grey', loc='left', style='italic')
plt.xticks(np.arange(min(contribution_size[:10]), max(contribution_size[:10])+1, 100), rotation = 45)          
plt.subplot(223)
plt.plot( contribution_size[20:30], bankruptcy_prob[20:30], marker='o', color="green", alpha=0.3)
plt.title('1000 powtórzeń symulacji', fontsize=12, color='grey', loc='left', style='italic')
plt.xlabel('Wysokość składki')
plt.ylabel('Prawdopodobieństwo bankructwa')
plt.xticks(np.arange(min(contribution_size[:10]), max(contribution_size[:10])+1, 100), rotation = 45)          
plt.subplot(224)
plt.plot(contribution_size[30:40], bankruptcy_prob[30:40], marker='o', color="orange", alpha=0.3)
plt.title('3000 powtórzeń symulacji', fontsize=12, color='grey', loc='left', style='italic')
plt.xlabel('Wysokość składki')
plt.xticks(np.arange(min(contribution_size[:10]), max(contribution_size[:10])+1, 100), rotation = 45)          
# Add a title:
plt.suptitle('Zależność przy różnej liczbie powtórzeń symulacji',fontsize=14, y=0.95)
plt.show()
plt.savefig("symulacje.pdf")

#B Maybe it would be worth to attract more customers?

#SIMULATION
avg_result = [] 
contribution_size = []
bankruptcy_prob = []
ruins_num = [] 
surplus_level = []
client_version = []

horizon = 2
n = 100

for client_num in range(10, 101, 10):
    for surplus in [10000]:
        for contribution in [900]:
            sim_output = func_call(surplus, contribution, n, 
                                   client_num, damages_avg, 
                                   damage_size_ln_avg, damage_size_ln_std, 
                                   horizon)
            client_version.append(client_num)
            surplus_level.append(surplus)
            contribution_size.append(contribution)
            ruins_num.append(sim_output[0])
            bankruptcy_prob.append(sim_output[1])
            avg_result.append(sim_output[2])
            print("Nadwyzka: ", surplus, "Skladka: ", contribution, 
            "Liczba ruin: ", sim_output[0], "Sredni wynik: ",
            round(sim_output[2]), "bankruptcy_prob: ", sim_output[1])
            
sns.set_style("whitegrid")
plt.plot(client_version[0:9], bankruptcy_prob[0:9], color="purple", marker="o")
plt.xticks(np.arange(min(client_version[0:9]), max(client_version[0:9])+1, 10))
plt.ylabel('Prawdopodobienstwo bankructwa')
plt.xlabel('Oczekiwana liczba klientów')
plt.title('Wpływ oczekiwanej liczby klientów na prawdopodobieństwo bankrutctwa', fontsize=14)
plt.show()

#B How about we start with a higher surplus? 

#SIMULATION
avg_result = [] 
contribution_size = []
bankruptcy_prob = []
ruins_num = [] 
surplus_level = []
client_version = []

horizon = 2
n = 100

for client_num in [100]:
    for surplus in  [10000, 15000, 30000]:
        for contribution in range(500,1101,100):
            sim_output = func_call(surplus, contribution, n, 
                                   client_num, damages_avg, 
                                   damage_size_ln_avg, damage_size_ln_std, 
                                   horizon)
            client_version.append(client_num)
            surplus_level.append(surplus)
            contribution_size.append(contribution)
            ruins_num.append(sim_output[0])
            bankruptcy_prob.append(sim_output[1])
            avg_result.append(sim_output[2])
            print("Nadwyzka: ", surplus, "Skladka: ", contribution, 
            "Liczba ruin: ", sim_output[0], "Sredni wynik: ",
            round(sim_output[2]), "bankruptcy_prob: ", sim_output[1])


area = [i * 0.02 for i in surplus_level]
sns.set_style("whitegrid")
plt.scatter(contribution_size[:7], bankruptcy_prob[:7], c=contribution_size[:7], s=area[:7], cmap=plt.cm.hsv, alpha = 0.5, label = '10000')
plt.scatter(contribution_size[7:14], bankruptcy_prob[7:14], c=contribution_size[7:14], s=area[7:14], cmap=plt.cm.hsv, alpha = 0.5, label = '15000')
plt.scatter(contribution_size[14:21], bankruptcy_prob[14:21], c=contribution_size[14:21], s=area[14:21], cmap=plt.cm.hsv, alpha = 0.5, label = '30000')
plt.xticks(np.arange(min(contribution_size), max(contribution_size)+1, 100))
plt.ylabel('Prawdopodobienstwo bankructwa')
plt.xlabel('Wysokosć składki')
plt.title('Wpływ nadwyżki początkowej na prawdopodobieństwo bankructwa', fontsize = 14)
plt.legend(loc='best',bbox_to_anchor=(1, 1), title = 'Nadwyżka:', facecolor = "whitesmoke",
           title_fontsize=11, edgecolor = "black", borderaxespad = 0, borderpad =1.75, labelspacing  =1.25, shadow=True)
plt.show()

to_map = pd.DataFrame({"Nadwyżka": surplus_level,"Składka": contribution_size, "Bankructwo": bankruptcy_prob})
to_map = pd.pivot_table(to_map, index='Nadwyżka', columns='Składka', values='Bankructwo', aggfunc=np.sum)
plt.figure(figsize=(10, 5))
sns.heatmap(to_map, annot=True, annot_kws={"size": 12}, cmap="BuPu", linewidths=0.5)
plt.yticks(rotation=90) 
plt.title("Mapa prawdopodobieństw dla trzech poziomów nadwyżki", fontsize = 15)
to_map.to_csv("C:/Users/Alicja Kocieniewska/Desktop/export_dataframe.csv", index=False, header=True)

#New idea: diversification of contributions - if clients like to risk - let's play some random game with them

#simulation model: 

seed = sc.random.seed(5) 

def model_random (client_num, damage_size_ln_avg, 
           damage_size_ln, damage_size_ln_std, horizon, 
           surplus, seed):
    contract_dates = [sc.random.randint(0, 364) for i in range(client_num)]
    calendar_in = [0]*365
    for contractDate in contract_dates:
        calendar_in[contractDate] += 1

    damages_for_client = sc.random.poisson(damages_avg, 
                                             client_num)
    contribution_list = [0] * client_num
    for k in range(client_num):
        contribution_list[k] = sc.random.randint(500, 2000)
    contribution = sc.mean(contribution_list)
    
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
            return [surplus - compensations, contribution_list]
        else:
            pass
        surplus -= compensations
    return [surplus, contribution_list]



#make it multiple: 
    
def func_call_random(surplus, n, client_num, damages_avg, damage_size_ln_avg, damage_size_ln_std, horizon):
    bankruptcy = 0
    positive_result = []
    result = []
    contribution_random = []
    for seed in range(n):
        result.append(model_random(client_num, damage_size_ln_avg, 
                             damage_size_ln, damage_size_ln_std, horizon, 
                             surplus, seed)[0])
        contribution_random.append(model_random(client_num, damage_size_ln_avg, 
                             damage_size_ln, damage_size_ln_std, horizon, 
                             surplus, seed)[1])
        if result[seed] < 0:
            bankruptcy += 1
        if result[seed] > 0:
            positive_result.append(result[seed])
    result_avg = sc.mean(positive_result)
    bankruptcy_prob = bankruptcy / n
    return [bankruptcy, bankruptcy_prob, result_avg, contribution_random]


#SIMULATION

horizon = 2
n = 100
client_num = 100
surplus = 10000

sim_output = func_call_random(surplus, n, client_num, damages_avg, 
                       damage_size_ln_avg, damage_size_ln_std, horizon)
contribution_track = sim_output[3]
ruins_num = sim_output[0]
bankruptcy_prob = sim_output[1]
avg_result = sim_output[2]
con_avg = []
for k in contribution_track:
    con_avg.append(sc.mean(k))
    
contribution_track = sum(contribution_track, []) #unlist the list 
sc.mean(contribution_track)
    
male = []
duze = []
for k in contribution_track[:100]:
    if k > np.mean(contribution_track[:100]):
        duze.append(k)
    else: 
        male.append(k)
        
male.sort()
duze.sort()
plt.figure(figsize=(15, 5))
plt.bar(range(50), male, color = 'green', width = 2, linewidth = 2, edgecolor = 'black')
plt.bar(range(50, 100), duze, color = 'red', width = 2, linewidth = 2, edgecolor = 'black')
plt.xticks(np.arange(0, 101, 2), rotation = 45)
plt.yticks(np.arange(0, 2001, 200))
plt.axhline(np.mean(contribution_track[:100]), color='blue', linewidth=4)
plt.xlabel('Numer klienta')
plt.ylabel("Wysokosc składki")
plt.title('Przykładowa iteracja przy losowanej składce', fontsize = 14)

sns.set_style("whitegrid")
plt.figure(figsize=(15, 5))
plt.bar(range(100), contribution_track[:100], color = 'skyblue', width = 1, linewidth = 2, edgecolor = 'black')
plt.axhline(np.mean(contribution_track[:100]), color='red', linewidth=4)
plt.xlabel('Numer klienta')
plt.ylabel("Wysokosc składki")
plt.title('Przykładowa iteracja przy losowanej składce', fontsize = 14)
min(contribution_track[:100])
max(contribution_track[:100])

plt.hist(con_avg, linewidth = 2, edgecolor = 'black', color = 'orange', bins=10)
plt.xlabel('Składka')
plt.ylabel("Liczba klientów")
plt.title('Histogram rozkładu składek w symulacji', fontsize = 14)
np.max(con_avg)

sns.distplot(con_avg, norm_hist=True, kde=True, bins=10, color = 'purple',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
plt.xlabel('Składka')
plt.ylabel("Udział procentowy")
plt.title('Przybliżony rozkład składek w symulacji', fontsize = 14)
test_if_normal = kstest(con_avg, sc.stats.norm.cdf, 
               args = (sc.mean(con_avg), sc.std(con_avg)))
if test_if_normal[1] > 0.05:
    print ("p-value wyniosło:", round(test_if_normal[1], 4), 
           "- brak więc podstaw do odrzucenia hipotezy " +
           "o normalnym rozkładzie zmiennej")
else:
    print ("należy odrzucic hipotezę zerowa")
























