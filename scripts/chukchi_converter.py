import re


cyrillic = 'пткмнлԓлврггччччсссйздбшф' #ч -> č is not correct for cyrlat
latin = 'ptkmnɬɬlwrɣgsčçcsçcjzdbšf'
symbols_cyr = ['к\'','н\'','к’','н’','ӄ','ӈ']
symbols_lat = 'qŋqŋqŋ'
jvowels_cyr = 'яеюё'
vowels_cyr = 'аэуоыи'
vowels_lat = 'aeuoəi'

cyrsymb = set(cyrillic + jvowels_cyr + vowels_cyr + ''.join(symbols_cyr))
latsymb = set(latin + vowels_lat + ''.join(symbols_lat))


def cyrlat(text):
    for i in range(len(symbols_cyr)):
        text = text.replace(symbols_cyr[i],symbols_lat[i])
        text = text.replace(symbols_cyr[i].upper(),symbols_lat[i].upper())
        
    for i in range(len(jvowels_cyr)):
        text = re.sub('[ъь]'+jvowels_cyr[i],'j'+vowels_lat[i],text)
        text = re.sub('\\b'+jvowels_cyr[i],'j'+vowels_lat[i],text)
        text = re.sub('\\b'+jvowels_cyr[i].upper(),'J'+vowels_lat[i],text)
        text = re.sub('([аэуоыАЭУОЫ])'+jvowels_cyr[i],'\\1j'+vowels_lat[i],text)
        text = text.replace(jvowels_cyr[i], vowels_lat[i])
        
    for i in range(len(vowels_cyr)):
        text = re.sub('[ъь]'+vowels_cyr[i],'ʔ'+vowels_lat[i],text)
        text = re.sub(vowels_cyr[i]+'[\'’]','ʔ'+vowels_lat[i],text)
        text = re.sub(vowels_cyr[i].upper()+'[\'’]','ʔ'+vowels_lat[i],text)
        text = text.replace(vowels_cyr[i],vowels_lat[i])
        text = text.replace(vowels_cyr[i].upper(),vowels_lat[i].upper())

    for i in range(len(cyrillic)):
        text = text.replace(cyrillic[i],latin[i])
        if cyrillic[i] != 'л' and cyrillic[i] != 'ԓ':
            text = text.replace(cyrillic[i].upper(),latin[i].upper())
        else:
            text = text.replace(cyrillic[i].upper(),latin[i])
    return text


def latcyr(text):
    text = re.sub('[çcs]q','сq',text) # cyrillic c
    
    for i in range(int(len(symbols_cyr) / 3)):
        text = text.replace(symbols_lat[i],symbols_cyr[i])
        text = text.replace(symbols_lat[i].upper(),symbols_cyr[i].upper())
        
    for i in range(len(jvowels_cyr)):
        text = text.replace('ʲ'+vowels_lat[i],jvowels_cyr[i])
        text = re.sub('\\bj'+vowels_lat[i],jvowels_cyr[i],text)
        text = re.sub('\\bJ'+vowels_lat[i],jvowels_cyr[i].upper(),text)
        text = re.sub('(['+vowels_lat+vowels_lat.upper()+jvowels_cyr+jvowels_cyr.upper()+vowels_cyr+vowels_cyr.upper()+'])j'+vowels_lat[i],
                      '\\1'+jvowels_cyr[i],text)
        text = re.sub('([ɬlçcs])j'+vowels_lat[i],'\\1ь'+jvowels_cyr[i],text)
        text = text.replace('([LÇCS])j'+vowels_lat[i],'\\1ь'+jvowels_cyr[i])
        text = text.replace('j'+vowels_lat[i],'ъ'+jvowels_cyr[i])
        text = re.sub('[ɬl]'+vowels_lat[i], 'л'+jvowels_cyr[i],text)
        text = text.replace('L'+vowels_lat[i], 'Л'+jvowels_cyr[i])
        
    text = re.sub('[cçs]e', 'че', text)
    text = re.sub('[CÇS]e', 'Че', text)

    for i in range(len(vowels_cyr)):
        text = text.replace('ʲ'+vowels_lat[i],vowels_cyr[i])
        text = re.sub('(['+vowels_lat+vowels_lat.upper()+'])[ʔˀɂ]'+vowels_lat[i],'\\1'+vowels_cyr[i]+'\'',text)
        text = re.sub('\\b[ʔˀɂ]'+vowels_lat[i],vowels_cyr[i]+'\'',text)
        text = re.sub('([ɬlçcs])[ʔˀɂ]'+vowels_lat[i],'\\1ь'+vowels_cyr[i],text)
        text = re.sub('([LÇCS])[ˀʔɂ]'+vowels_lat[i],'\\1ь'+vowels_cyr[i],text)
        text = re.sub('[ˀʔɂ]'+vowels_lat[i],'ъ'+vowels_cyr[i],text)
        text = text.replace(vowels_lat[i],vowels_cyr[i])
        text = text.replace(vowels_lat[i].upper(),vowels_cyr[i].upper())

    text = text.replace('ʲ','ь')
    
    for i in range(len(cyrillic)):
        text = text.replace(latin[i],cyrillic[i])
        text = text.replace(latin[i].upper(),cyrillic[i].upper())
        
    return text


def converter(query):
    ncyr = 0
    nlat = 0
    for symb in query:
        if symb in cyrsymb:
            ncyr += 1
        elif symb in latsymb:
            nlat += 1
    if ncyr > nlat:
        transliterated = cyrlat(query)
    else:
        transliterated = latcyr(query)
    return transliterated
    
    
if __name__ == '__main__':
    with open('input.txt','r',encoding='utf-8-sig') as f1:
        text = f1.read()
    f = open('output.txt','w',encoding='utf-8-sig')
    transl = converter(text)
    f.write(transl)
    f.close()
