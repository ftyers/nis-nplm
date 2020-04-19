# Corpora files
## Corpus
### corpus_v1.txt
Is written using Cyrillic transliteration; it contains all the data from chukchi novels and periodicals 
### corpus_v2.txt
Is made from "corpus_v1.txt" using the first script from Processing steps
### corpus_for_segmentation
Is the input data for the segmentation model
### coupus.out
Is the output of the segmentation model
### corpus_as_standard
Is the "coupus.out" converted using the scripts/corpus_converter.py

## Gold standard
### eng_standard
Is written using Latin transliteration; it contains the data which is used as the gold standart 
for segmentation; all the data from here can be found here: http://chuklang.ru/full_texts
### ru_standard
Is written using Cyrillic transliteration; it was transliterated using http://chuklang.ru/converter and 
then manually reviewed
### ru_standard_v2
Is made from "ru_standart" using the first script from Processing steps
### ru_standard_new_part
Is the manually checked data from "corpus_as_standard" to be added to the gold standart

Describe what the splits are you are using (or are you doing cross-validation)

Describe what accuracy you get for the model


# Processing steps
- fix apostrophes after "н" and "к"
```
sed "s/к['’]/ӄ/g" | sed "s/н[’']/ӈ/g" | sed "s/К['’]/Ӄ/g" | sed "s/Н['’]/Ӈ/g" 
```

