# Decision-support-in-dermatology

This repository contains the scripts used in "Deep learning for decision support in dermatology". This work was presented as a M. Sc. Thesis for the Technical University of Denmark (DTU).

## Getting Started

Before starting download doadload the data and place it in this structure:

 <img src="https://github.com/kalilamali/Decision-support-in-dermatology/blob/master/data_structure.png" width="200" height="200">

### Prerequisites

See requirements.txt

### Experiments

A step by step series of examples that tell you how to get a development env running

First downsampled the data

```
resize_all_data.py
```

Go to the folder of the experiments you want to run.

```
cd binary
build_data.py
train.py
evaluate.py
```

### Questions

For questions, contact s181423@student.dtu.dk

*Related work* - [SKINdx](https://github.com/kalilamali/SKINdx)

## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
