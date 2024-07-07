# Named Entity Recognition on German commercial register documents

This project deals with the recognition of Named Entities in documents of the German commercial register. 

## Description

This project explores Named Entity Recognition (NER) in documents from the German commercial register, made accessible following a decision by the European Court of Justice. Various Language Models were trained for NER, utilizing specific and general annotations tailored to the register's documents. Prior to training, comprehensive data analysis and annotation were conducted to establish a gold standard. The findings highlight the crucial role of data quality in NER performance, emphasizing the significance of thorough annotation. Surprisingly, the choice of Language Model proved less decisive than anticipated, with the data itself emerging as the primary determinant of performance. Notably, models trained on more general annotations consistently outperformed those trained on specific ones which is supported by significant results from a Wilcoxon signed-pair test. To ensure the robustness of the findings, a Cross-Validation procedure was employed, guarding against chance learning or overfitting. Through this approach,the reliability of the results is affirmed and underscore the importance of thorough data preparation in NER tasks.

The data utilized in this master's thesis is not available for public access due to confidentiality reasons, as it was obtained from proprietary sources.

## Getting Started

### Dependencies
For a basic usage of training and evaluating models: 

* models were trained with the Python library spaCy. To install it:
  ```
  pip install -U spacy
  ```
* the large Language Model provided by spaCy on which the training is based on:
  ```
  python -m spacy download de_core_news_lg
  ``` 

### Installing

* Clone repository via:
  ```
  git@github.com:danieltepavac/ner_handelsregister.git
  ```
* Then it can be already used if dependencies are installed.

### Executing program

* Depending on the data, run `data_transformation.py` to transform your data in the correct spaCy-Format.
* Train your data with `train.py`. Select the function you want to use and define the source and save paths. Functions use different language models as foundation:
  * `def train_blank` uses the default German language model provided by spaCy.
  * `def train_german_lm` uses `de_core_news_lg`.
  * `def train_transformer`uses `de_core_news_lg` as well but adds a Transformer pipeline to the process. `bert-base-german-cased` is used here.
* If there is validation data, use the respective function with the extension `_val`
* For a quick visual evaluation use `spacy_test_0.py`. For a more thorough evaluation, use `evaluate.py`. 


## Authors

Contributors names and contact info

ex. Daniel Tepavac 


## Version History

* 0.1
    * Initial Release
