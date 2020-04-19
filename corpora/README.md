# Corpora files
There are several files with corpora data:
- "corpus_v1.txt" is written using Cyrillic transliteration; it contains all the data from chukchi novels and journals 
- "corpus_v2.txt" is made from "corpus_v1.txt" using the first script from Processing steps
- "eng_standart" is written using Latin transliteration; it contains the data which is used as the gold standart 
for segmentation; all the data from here can be found here: http://chuklang.ru/full_texts
- "ru_standart" is written using Cyrillic transliteration; it was transliterated using http://chuklang.ru/converter and 
then manually reviewed
- "ru_standart_v2" is made from "ru_standart" using the first script from Processing steps
- "corpus_for_segmentation" is the input data for the segmentation model
- "coupus.out" is the output of the segmentation model
- "corpus_as_standart" is the "coupus.out" converted using the scripts/corpus_converter.py
- "ru_standart_new_part" is the manually checked data from "corpus_as_standart" to be added to the gold standart

Describe what the splits are you are using (or are you doing cross-validation)

Describe what accuracy you get for the model


# Processing steps
- fix apostrophes after "н" and "к"
```
sed "s/к['’]/ӄ/g" | sed "s/н[’']/ӈ/g" | sed "s/К['’]/Ӄ/g" | sed "s/Н['’]/Ӈ/g" 
```

