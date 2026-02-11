## author

Romeo De Timmerman

## project summary

This study examines the stability of African American English (AAE) features in blues lyrics across different performance contexts. Using a corpus of 540 songs, each performed both live and in the studio by 45 artists from diverse sociocultural backgrounds and time periods, the study annotates eight phonological and lexico‑grammatical AAE features in 30,000+ tokens. Through descriptive statistics and gradient‑boosted decision tree modeling with SHAP explainability, the results show that AAE features are consistently employed in blues performance, with minimal systematic differences between live and studio recordings, song types, time periods, or artist backgrounds. Instead, AAE realization is primarily driven by linguistic factors such as segmental context and lexical frequency. The findings support the argument that AAE features function as robust stylistic conventions in the blues, viz. as indexical of artistic authenticity, and increasingly iconic of the genre itself.

## project structure

```
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Documents
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-rdt-initial-data-exploration`.
│
├── figures            <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py

(this project structure is based on the Cookiecutter Data Science template)
```
