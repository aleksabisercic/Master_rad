# Aleksa Bisercic Master Thesis: A novel methodology for simulation of industrial systems

## Abstract 
*With the increase in the maintenance of mining systems predicting failure rates became an essential aspect in optimizing the maintenance performance. Despite the increase of computational power, there is no significant development in improving the simulation and prediction of avalibilty and reliability of industrial systems. With this problem at hand machine learning algorithms, being an increasingly more important part of everyday life, have potential to become one of the key ingredients in simulation and prediction within the idustrial systems.. Therefore in this paper we introduce a novel methodology for offline and online simulation availibity and reliability prediction of industrial and mining systems. Whereas traditional methodology for simulating availability and reliability is a widely used industry standard statistical and probabilistic model that implements Monte Carlo method, our proposed methodology for simulating such a system is divided in three subsegments : 1. using deep learning to generate failure intensity together with an inhomogeneous poisson point process to  predict the number and probabilistic time of failure; 2. using various neural network architectures to predict length of repair (after failure) and time between failures; and lastly 3. using LSTM encoder-decoder network to classify types of failures that will happen. Taking the industrial mining system (Excavator-Belt Conveyor-Crusher EBC system) as a case study, we evaluate our proposed methodology on three highly competitive benchmark tasks: accuracy of prediction for failure and repair rate, length of failures and lastly accuracy of classification. Furthermore, the presented methodology is not limited to mining systems and can be implemented to other industrial systems, where failure and maintenance rate are applied and can, to the best of our knowledge, be used to further optimize proactive maintenance strategies.*

## Methodology Algorithm (currently in Serbian)
 +![FlowChart Metodologije - Frame](https://user-images.githubusercontent.com/64646644/110998369-ff766800-837e-11eb-9b5d-774c21f6a227.jpg)

## Case study used: 
### Simbolic diagram of Indastrial system used
![BTD_sistem](https://user-images.githubusercontent.com/64646644/112521605-fcc33c00-8d9c-11eb-91eb-49cfbfb26bca.jpg)
*Where M is mechanical component, E is electrical component and O are other component of the respective systems.*

### Simulation Algorithm (currently in Serbian)
![FlowChart Statisticke Sim](https://user-images.githubusercontent.com/64646644/110998430-161cbf00-837f-11eb-9f9f-4ab1fee899fa.jpg)

### GeoMap of the system
![kop](https://user-images.githubusercontent.com/64646644/112521957-62172d00-8d9d-11eb-8030-a6236a4bf487.png)

## LSTM architecture chosen 
![LSTM Encoder-decoder Multiver timeseries graph](https://user-images.githubusercontent.com/64646644/112522252-b3272100-8d9d-11eb-9066-d6d0b518e047.png)

## Results:
Prediction of failure rates: **improvment of over 30% above widly used industrial standard methodology**: 
![Capture11](https://user-images.githubusercontent.com/64646644/110999733-1ddd6300-8381-11eb-9175-499077b5ce67.PNG)
