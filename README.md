This project deals with the recognition of Named Entities in documents of the German commercial register. 
It was carried out as part of the master's thesis "Named Entitiy Recognition on German commercial register documents" at the University of Regensburg. 

Abstract: <br>
This study explores Named Entity Recognition (NER) in documents from the German commercial register, made accessible following a decision by the European Court of Justice. Various Language Models were trained for NER, utilizing specific and general annotations tailored to the register's documents. Prior to training, comprehensive data analysis and annotation were conducted to establish a gold standard. The findings highlight the crucial role of data quality in NER performance, emphasizing the significance of thorough annotation. Surprisingly, the choice of Language Model proved less decisive than anticipated, with the data itself emerging as the primary determinant of performance. Notably, models trained on more general annotations consistently outperformed those trained on specific ones which is supported by significant results from a Wilcoxon signed-pair test. To ensure the robustness of the findings, a Cross-Validation procedure was employed, guarding against chance learning or overfitting. Through this approach,the reliability of the results is affirmed and underscore the importance of thorough data preparation in NER tasks.

Usage: <br>
In data_transformation.py there are functions provided transforming the data into the necessary spaCy format. 
Then, it can be trained with the train.py script. <br>
spacy_test_0.py offers a possibilty to check a concrete example, on how the model has worked. 

Proving the validity of the models are tests conducted in the realm of an evaluation, Cross-Validation, Wilcoxon signed-paired test. There are respective scripts. 

Results: <br>
All images and information on which the tables are built on in the Master Thesis are saved here as well. 

Data: <br>
The data utilized in this master's thesis is not available for public access due to confidentiality reasons, as it was obtained from proprietary sources. However, a detailed summary of the dataset used in this research, including its structure, characteristics, and key findings derived from the data analysis, is provided within the thesis document. This summary aims to offer insights into the research methodology and results.
