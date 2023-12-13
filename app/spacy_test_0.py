# Import spacy and displacy.
import spacy
from spacy import displacy 

# First try: Load German LM. 
NER = spacy.load("de_core_news_lg")


# This is the example text.
example = "Urkundenrolle 493 \/ 2018 W UZ 1345\/2018 W Notarkanzlei Kurz Johann-Gottfried-Pahl-Straße 4 73430 Aalen Amtsgericht Ulm - Registergericht --- Zeughausgasse 14 89073 Ulm HRB 501670 Firma GUIRAUD GmbH mit Sitz in Ulm Handelsregisteranmeldung I. Zur Eintragung in das Handelsregister wird angemeldet: Zum Geschäftsführer der Gesellschaft wurde bestellt: Herr Andreas Kirsch, geboren am 17.04.1962, wohnhaft Talstraße 45\/1 in 89195 Staig, Die konkrete Vertretungsregelung lautet: Andreas Kirsch vertritt satzungsgemäß und ist von den Beschränkungen des § 181 BGB 2. Alt. befreit. Die Satzung wurde in § 3 Absatz (3) (Stammkapital, Dauer der Gesellschaft und Geschäftsjahr) hinsichtlich des Geschäftsjahres geändert. II. Ich, der Unterzeichner, versichere nach notarieller Belehrung über die Strafbarkeit einer falschen Versicherung, dass meiner Bestellung keine Umstände nach § 6 Abs. 2 Satz 2 Nr. 2 und 3 sowie Satz 3 GmbHG entgegenstehen, d. h., es erfolgte im Inland keine rechtskräftige Verurteilung - maßgebend ist der Eintritt der Rechtskraft der Entscheidung - wegen einer oder mehrerer vorsätzlich begangener"

# Apply NER on example.
NER_example = NER(example)

# Show each recognized entity + its label.
for word in NER_example.ents:
    print(word.text, word.label_)

# Show how the text is marked up. In Browser: 
# displacy.serve(NER_example, style="ent")

# Show how the text is marked up. In Notebook:
displacy.render(NER_example, style="ent")  

# Result: It does work but not every recognized entity is an entity