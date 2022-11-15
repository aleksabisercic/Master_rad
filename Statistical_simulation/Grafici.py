import numpy as np
import matplotlib.pyplot as plt
import pickle

name = "BTD"
broj_sek_plot = 2000000
sample = [150,45,33]


with open('Rezultati/vremena_otkaza_{}.pkl'.format(name), 'rb') as f1:
    vremena_otkaza = pickle.load(f1)

with open('Rezultati/vremena_popravki_{}.pkl'.format(name), 'rb') as f2:
    vremena_popravke = pickle.load(f2)

vremena_otkaza = np.squeeze(vremena_otkaza)


def izracunaj_y(broj_sek_plot,vremena_otkaza,vremena_popravke):
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
    return x, y1


lista_x = []
lista_y = []

figure1 = plt.figure(figsize=(13, 9))

ax1 = plt.subplot(311)
ax2 = plt.subplot(312)
ax3 = plt.subplot(313)

for i in range(3):
    s = sample[i]
    x, y = izracunaj_y(broj_sek_plot, vremena_otkaza[s], vremena_popravke[s])
    lista_x.append(x)
    lista_y.append(y)

ax1.fill_between(lista_x[0], lista_y[0], step="pre", alpha=0.4)
ax1.step(lista_x[0], lista_y[0], label='pre (default)')
ax2.fill_between(lista_x[1], lista_y[1], step="pre", alpha=0.4)
ax2.step(lista_x[1], lista_y[1], label='pre (default)')
ax3.fill_between(lista_x[2], lista_y[2], step="pre", alpha=0.4)
ax3.step(lista_x[2], lista_y[2], label='pre (default)')

ax2.set_ylabel("Stanje sistema")
ax1.set_ylabel("Stanje sistema")
ax3.set_xlabel("Vreme(t)")
ax3.set_ylabel("Stanje sistema")
ax1.set_xlim(0,broj_sek_plot)
ax2.set_xlim(0,broj_sek_plot)
ax3.set_xlim(0,broj_sek_plot)




# figure1.title("Rad sistema")


plt.savefig('Slike/simulacija.png')
plt.show()	