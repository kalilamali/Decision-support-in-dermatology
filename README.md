# Decision-support-in-dermatology

This repository contains the scripts used in "Deep learning for decision support in dermatology". This work was presented as a M. Sc. Thesis for the Technical University of Denmark (DTU).

## Getting Started

Before starting download doadload the data and place it in this structure:

 ![image](https://github.com/kalilamali/Decision-support-in-dermatology/blob/master/data_structure.png){ width=10% height=10% }

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
