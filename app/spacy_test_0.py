# Import spacy and displacy.
import spacy
from spacy import displacy 

# First try: Load German LM. 
NER = spacy.load("de_core_news_lg")


# This is the example text.
example = "Wir sind die alleinigen Gesellschafter der im Handelsregister des Amtsgerichts Köln unter HR B 81000 eingetragenen mit Sitz in Köln. Gesellschafterbeschluss First Contor Immobilien GmbH Unter Verzicht auf alle gesetz- und satzungsmäßigen Frist- und Formvorschriften halten wir hiermit eine Gesellschafterversammlung ab und beschließen einstimmig wie folgt: 등 (Dominik Recke) Herr Ismail C in a r, geboren am 01. November 1964, wohnhaft Scharpenburger Straße 24 A, 58256 Ennepetal, wird zum weiteren Geschäftsführer bestellt. Er ist einzelvertretungsbefugt und von den Beschränkungen des § 181 BGB befreit. Weiteres war nicht zu beschließen. Köln, den 08. Juni 2020 سکا اردو کے (Ismail Cinar)"

# Apply NER on example.
NER_example = NER(example)

# Show each recognized entity + its label.
for word in NER_example.ents:
    print(word.text, word.label_)

# Show how the text is marked up. In Browser: 
displacy.serve(NER_example, style="ent")

# Show how the text is marked up. In Notebook:
displacy.render(NER_example, style="ent")  

# Result: It does work but not every recognized entity is an entity