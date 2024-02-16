# Import spacy and displacy.
import spacy
from spacy import displacy 

from pathlib import Path

# First try: Load German LM. 
NER = spacy.load("de_core_news_lg")

model_dir2 = Path(Path(__file__).parents[1], "../models/cross_validation/d1/best_model_fold_1.spacy")

NER2 = spacy.load(model_dir2)


# This is the example text.
example = "Wir sind die alleinigen Gesellschafter der im Handelsregister des Amtsgerichts Köln unter HR B 81000 eingetragenen mit Sitz in Köln. Gesellschafterbeschluss First Contor Immobilien GmbH Unter Verzicht auf alle gesetz- und satzungsmäßigen Frist- und Formvorschriften halten wir hiermit eine Gesellschafterversammlung ab und beschließen einstimmig wie folgt: 등 (Dominik Recke) Herr Ismail C in a r, geboren am 01. November 1964, wohnhaft Scharpenburger Straße 24 A, 58256 Ennepetal, wird zum weiteren Geschäftsführer bestellt. Er ist einzelvertretungsbefugt und von den Beschränkungen des § 181 BGB befreit. Weiteres war nicht zu beschließen. Köln, den 08. Juni 2020 سکا اردو کے (Ismail Cinar)"
example2 = "Amtsgericht Jena - Handelsregister - Rathenaustraße 13 07745 Jena HRB 112399 P.H. Personal-Gesellschaft für Zeitarbeit mbH Zur Eintragung in das Handelsregister wird angemeldet: Zum weiteren Geschäftsführer wurde bestellt: Herr Jens Hoffmann, geboren am 4. Mai 1965, wohnhaft Martin-Andersen-Nexö-Straße 34 in 99096 Erfurt. Er ist stets allein zur Vertretung der Gesellschaft berechtigt, auch wenn mehrere Geschäftsführer bestellt sind. Er ist von den Beschränkungen des § 181 BGB befreit. Nach Belehrung durch den Notar über die unbeschränkte Auskunftspflicht gegenüber dem Gericht gemäß § 53 Absatz (1) und Absatz (2) des Ge- setzes über das Zentralregister und das Erziehungsregister und die Straf- barkeit einer falschen Versicherung (§ 8 GmbHG) versichert der Ge- schäftsführer, dass: keine Umstände vorliegen, aufgrund deren der Geschäftsführer nach § 6 GmbHG von dem Amt als Geschäftsführer ausgeschlossen wäre: Während der letzten 5 Jahre erfolgte keine Verurteilung (also auch keine rechtskräftige) nach §§ 283 bis 283d StGB (Insolvenzstrafta- ten), der falschen Angaben nach § 82 GmbHG oder § 399 des Akti- engesetzes, der unrichtigen Darstellung nach § 400 des Aktienge- setzes, § 331 des Handelsgesetzbuchs, § 313 des Umwandlungsge-"
example3 = "Nr. 143 der Urkundenrolle Jahrgang 2019 Vor mir, dem unterzeichnenden Verhandelt zu Hannover, am 10. Mai 2019 erschien heute: Notar Dr. Nikolas v. Wrangell mit dem Amtssitz in 30175 Hannover, Adenauerallee 8, Frau Maren Fischer, geb. am 16.01.1969, wohnhaft Örtzeweg 8, 31303 Burgdorf, ausgewiesen durch gültigen Bundespersonalausweis, handelnd nicht für sich persönlich, sondern in ihrer Eigenschaft als einzelvertre- tungsberechtigte und von den Beschränkungen des § 181 BGB befreite Geschäfts- führerin der Gessert Verwaltungs GmbH, mit dem Sitz in Burgdorf, eingetragen im Handelsregister des Amtgerichts Hildesheim unter HRB 22640, diese wiederum han- delnd für die Gessert GmbH & Co. KG., mit dem Sitz in Burgdorf eingetragen im Handelsregister des Amtsgerichts Hildesheim unter HRA 21264. Der amtierende Notar bescheinigt aufgrund Einsichtnahme vom 08.05.2019 in das elektronisch geführte Handelsregister des Amtsgerichts Hildesheim zu HRB 22640, dass die Erschienene dort als einzelvertretungsberechtigte und von den Beschrän- kungen des § 181 BGB befreite Geschäftsführerin eingetragen ist. Der amtierende Notar bescheinigt weiterhin aufgrund Einsichtnahme vom 08.05.2019 in das elektro- nisch geführte Handelsregister des Amtsgerichts Hildesheim zu HRA 21264, dass die Vertretene zu 1. dort als persönlich haftende Gesellschafterin eingetragen ist. \/ 2"

# Apply NER on example.
NER_example = NER(example3)

# Show each recognized entity + its label.
for word in NER_example.ents:
    print(word.text, word.label_)

# Show how the text is marked up. In Browser: 
displacy.serve(NER_example, style="ent")


# Show how the text is marked up. In Notebook:
displacy.render(NER_example, style="ent")  

# Result: It does work but not every recognized entity is an entity.