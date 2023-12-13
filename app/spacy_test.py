
# Import spacy and displacy.
import spacy
from spacy import displacy 

from pathlib import Path

# First try: Load German LM. 

model_directory = Path(Path(__file__).parent, "ner")
print(model_directory)

NER = spacy.load(model_directory)


# This is the example text.
example = "Anlage zur Niederschrift vom 09.02.2015Urkundenrolle Nr. 90/2015 des NotarsDr. Thorsten Mätzig in DortmundGesellschaftsvertragderiVita GmbH"
example2 = "URNr:An das AmtsgerichtRegistergericht83276 TraunsteinM 3089/2021Az: 21319015Sb: ps/orHRB 14083EMM Einwanger Metall und Montage GmbHSitz: WinhöringGeschäftsadresse: Trenbeckstraße 20, 84543 WinhöringZur Eintragung in das Handelsregister wird angemeldet:Der Wohnort des Geschäftsführers Tobias Einwanger ist nun-mehr 84567 Perach.Die Geschäftsanschrift ergibt sich aus der Kopfzeile.Altötting, den 2. 12. 21Tobias Filening er1HZTE"

# Apply NER on example.

NER_example = NER(example2)

# Show each recognized entity + its label.
for word in NER_example.ents:
    print(word.text, word.label_)

# Show how the text is marked up. In Browser: 
# displacy.serve(NER_example, style="ent")

# Show how the text is marked up. In Notebook:
displacy.render(NER_example, style="ent")  

# Result: It does work but not every recognized entity is an entity