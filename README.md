# ExoRV

## Quickly examine RV information:

Plot:
  1. RV data
  2. Expected RV curve
  3. Both: RV data against an expected curve

Capabilities:
- Easily download planet parameters from ExoFOP
- Use Chen & Kipping (2016) to calculate an expected mass based on a transiting planet radius -- or input your own mass value
- Calculate the RMS of data points against the expected curve
- Compare RV datapoints from multiple datasets
- Implement known instrumental offsets


## Installation:

```
git clone https://github.com/brownn11/ExoRV.git
cd ExoRV
pip install -e .
```

## Running:

```
cd exorv
python main_maroonx.py -TOI 1827 -p 'tutorial/' -tn 'Gl 486_set1' --compare 'Gl 486_set2, Gl 486_set3'
```

![tutorial](https://github.com/user-attachments/assets/bf60c684-4f3d-44a6-b21d-137b46dd297c)

## Dependencies: 

- numpy
- matplotlib
- astropy
- pandas
- pathlib


