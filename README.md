# kNN_BPP
This repository contains the implementation of the kNN_BPP algorithm from the paper *Handling Class Imbalance in k-Nearest Neighbor Classification by Balancing Prior Probabilities* by Jonatan M. N. Gøttcke and A. Zimek.
The paper is presented at the *14th International Conference on
Similarity Search and Applications*, **SISAP 2021**. 

Furthermore the repository contains the implementations of the DIRECT CS kNN presented in the 2013 paper *Cost-Sensitive Classification with k-Nearest Neighbors* and the 2020 paper *Cost-sensitive KNN classification* by Qin. et al. and Zhang et al. 

And the CW-kNN algorithm from the *Class Based Weighted K-Nearest Neighbor over Imbalance Dataset* by V. Pudi and H. Dubey is implemented. 

 

The implementations follow the Scikit-learn classifier style which most users are familiar with. 
## Using the implementation 
1. Git clone the repository. 
2. Ensure you have Python 3.8.+ installed 
3. Install the required dependencies
    ```bash
    pip install -r requirements.txt
    ```
4. Try the example
    ```bash
    python example.py
    ```


See the example.py file on how to use the implementation in further detail. 

## Citation
```{r, eval=True}
@inproceedings{DBLP:conf/sisap/GottckeZ21,
  author    = {Jonatan M{\o}ller Nuutinen G{\o}ttcke and
               Arthur Zimek},
  title     = {Handling Class Imbalance in k-Nearest Neighbor Classification by Balancing
               Prior Probabilities},
  booktitle = {{SISAP}},
  series    = {Lecture Notes in Computer Science},
  volume    = {13058},
  pages     = {247--261},
  publisher = {Springer},
  year      = {2021}
}
``` 

### DOI 
```{r, eval=True}
DOI: 10.1007/978-3-030-89657-7_19
``` 