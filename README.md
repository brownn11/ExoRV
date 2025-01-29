# ExoRV

Plot (1) RV data on its own, (2) against an expected RV curve, or (3) only the RV curve.

--> Easily download planet parameters from ExoFOP

-->--> Use Chen & Kipping (2016) to calculate an expected mass based on a transiting planet radius -- or input your own mass value

--> Calculate the RMS of data points against the expected curve

--> Compare RV points of up different datasets



Future fixes:

--> Is poorly adapted for non-MAROON-X data

-->--> Currently only functioning for 2-instrument data products with endings '_r.csv' and '_b.csv', and naming conventions like 'Barnard_r.csv', 'Barnard_b.csv'

-->--> general scp only downloads one folder at a time

Example:

I reduced some Gl 486 data with different calibration files, and wanted to compare the results. I called: 

<python main_maroonx.py -TOI 1827 -p 'tutorial/' -tn 'Gl 486_set1' --compare 'Gl 486_set2, Gl 486_set3'>

And got the following plot:

![tutorial](https://github.com/user-attachments/assets/bf60c684-4f3d-44a6-b21d-137b46dd297c)
