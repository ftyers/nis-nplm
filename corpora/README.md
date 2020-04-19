# Corpora files
## Versioning
### v1
Initial data
### v2
Data with some of the apostrophes fixed (after "н" and "к")
### v3
Data without some of the unnecessary characters and with character fixes (meaning turning "C" (ord == 67, English) into "С" (ord == 1057, Russian))
### v4
Data with all the apostrophes fixed - if there is one after the vowel, turn it into "ʔ" and put it before the vowel; other case - remove it

## Corpus
### corpus_v1.txt
Is written using Cyrillic transliteration; it contains all the data from chukchi novels and periodicals 
### corpus_v2.txt
Is made from "corpus_v1.txt" using the first script from Processing steps
### corpus_v3.txt
Is made from "corpus_v2.txt" using the scripts/corpus_cleaner.py find_bad_strings func and the manual review of bad strings
### corpus_v4.txt
Is made from "corpus_v3.txt" using the scripts/corpus_cleaner.py fix_apostrophes func
### corpus_for_segmentation
Is the input data for the segmentation model (from corpus_v1.txt)
### coupus.out
Is the output of the segmentation model (from corpus_v1.txt)
### corpus_as_standard
Is the "coupus.out" converted using the scripts/corpus_converter.py

## Gold standard
### eng_standard
Is written using Latin transliteration; it contains the data which is used as the gold standard
for segmentation; all the data from here can be found here: http://chuklang.ru/full_texts
### ru_standard
Is written using Cyrillic transliteration; it was transliterated using http://chuklang.ru/converter and 
then manually reviewed
### ru_standard_v2
Is made from "ru_standard" using the second script from Processing steps
### ru_standard_v3
Is made from "ru_standard_v2" using the scripts/corpus_cleaner.py find_bad_strings func
### ru_standard_v4
Is made from "ru_standard_v3" using the scripts/corpus_cleaner.py fix_apostrophes func
### ru_standard_new_part
Is the manually checked data from "corpus_as_standard" to be added to the gold standard

Describe what the splits are you are using (or are you doing cross-validation)

Describe what accuracy you get for the model


# Processing steps
- fix apostrophes after "н" and "к"
```
sed "s/к['’]/ӄ/g" | sed "s/н[’']/ӈ/g" | sed "s/К['’]/Ӄ/g" | sed "s/Н['’]/Ӈ/g" 
```
- fix apostrophes after "н" and "к" for standard
```
sed "s/к['’]/ӄ/g" | sed "s/н[’']/ӈ/g" | sed "s/К['’]/Ӄ/g" | sed "s/Н['’]/Ӈ/g" | sed "s/к>'/ӄ>/g" | sed "s/н>'/ӈ>/g"
```
