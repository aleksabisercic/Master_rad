import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('Rezultati/vremena_otkaza.pkl', 'rb') as f1:
    vremena_otkaza = pickle.load(f1)

with open('Rezultati/vremena_popravki.pkl', 'rb') as f2:
    vremena_popravke = pickle.load(f2)

broj_sek_plot = 500000
sample = 0

vremena_otkaza = vremena_otkaza[sample]
vremena_otkaza = np.squeeze(vremena_otkaza)
vremena_popravke = vremena_popravke[sample]


x = np.arange(broj_sek_plot)
y1 = []
br = 0
for t in range(broj_sek_plot):
    
    if br < len(vremena_otkaza)-1:
        if t >=  vremena_otkaza[br] and t < vremena_popravke[br]:
            y = 0
        else:
            y = 1
        if t == vremena_popravke[br]:
            br = br + 1
    y1.append(y)

y1 = np.asarray(y1)

plt.fill_between(x, y1, step="pre", alpha=0.4)
plt.step(x, y1, label='pre (default)')
# plt.plot(x, y1, 'C0o', alpha=0)

plt.xlabel("Vreme(t)")
plt.ylabel("Stanje sistema")
plt.title("Rad sistema")

plt.show()	