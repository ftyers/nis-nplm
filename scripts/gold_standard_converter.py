import re
from bs4 import BeautifulSoup


def check_word(full, morphemes):
    segm_full = ''
    for m in morphemes:
        segm_full += re.sub('∅|\*|=', '', m.text.strip().strip('-'))
    return segm_full == full.text.strip()


def read_incorrect():
    res = []
    with open('/home/qb/Desktop/UD_Chukchi-Amguema/not-for-release/chukchi_texts_20181227.htm') as f:
        t = BeautifulSoup(f.read(), 'lxml')
        for child in t.select('.itx_Words'):
            new_l = []
            not_take = False
            for word in child.select('.itx_Frame_Word'):
                full = word.select('.itx_txt')
                morphemes = word.select('.itx_morph_txt')
                segm_full = ""
                if len(full) == 0 or len(morphemes) == 0:
                    continue
                if not check_word(full[0], morphemes):
                    not_take = True
                    segm_full = full[0].text.strip() + ' ('
                for m in morphemes:
                    segm_full += re.sub('\*|=', '', m.text.strip().strip('-')) + '>'
                segm_full = segm_full.rstrip('>')
                if not check_word(full[0], morphemes):
                    segm_full += ')'
                new_l.append(segm_full)
            if not_take and len(new_l) != 0:
                res.append(' '.join(new_l))
    return res


def read():
    res = []
    with open('/home/qb/Desktop/UD_Chukchi-Amguema/not-for-release/chukchi_texts_20181227.htm') as f:
        t = BeautifulSoup(f.read(), 'lxml')
        for child in t.select('.itx_Words'):
            new_l = []
            not_take = False
            for word in child.select('.itx_Frame_Word'):
                full = word.select('.itx_txt')
                morphemes = word.select('.itx_morph_txt')
                if len(full) == 0 or len(morphemes) == 0:
                    continue
                if check_word(full[0], morphemes):
                    segm_full = ''
                    for m in morphemes:
                        segm_full += re.sub('\*|=', '', m.text.strip().strip('-')) + '>'
                    new_l.append(segm_full.rstrip('>'))
                else:
                    not_take = True
            if not not_take and len(new_l) != 0:
                res.append(' '.join(new_l))
    return res


def write(res, file):
    with open(file, 'w') as f:
        f.write('\n'.join(res))


def split(file, mask_file, partial_file):
    with open(file) as origin:
        mask = open(mask_file, "w")
        partial = open(partial_file, "w")
        lines = origin.readlines()
        for line in lines:
            for symbol in line:
                if symbol == ">" or symbol == "∅":
                    mask.write(symbol)
                elif symbol == " " or symbol == "\n":
                    partial.write(symbol)
                    mask.write(symbol)
                else:
                    partial.write(symbol)
                    mask.write("_")
        mask.close()
        partial.close()


def combine(mask_file, partial_file, result):
    with open(result, "w") as result:
        mask = open(mask_file)
        partial = open(partial_file)
        mask_lines = mask.readlines()
        partial_lines = partial.readlines()
        if len(mask_lines) != len(partial_lines):
            print("different lengths")
            return
        for i in range(len(mask_lines)):
            res = ""
            masked = mask_lines[i]
            partialed = partial_lines[i]
            m, p = 0, 0
            try:
                while True:
                    if m >= len(masked) and p >= len(partialed):
                        break
                    elif masked[m] == ">" or masked[m] == "∅":
                        res += masked[m]
                        m += 1
                    elif (masked[m] == " " and partialed[p] == " ") or (masked[m] == "\n" and partialed[p] == "\n"):
                        res += masked[m]
                        m += 1
                        p += 1
                    else:
                        res += partialed[p]
                        p += 1
                        m += 1
                result.write(res)
            except IndexError:
                result.write("!! " + partialed)
        mask.close()
        partial.close()


def split_file(standart1, standart2, n):
    lines1 = open(standart1).readlines()
    lines2 = open(standart2).readlines()
    if len(lines1) != len(lines2):
        print("different file lengths")
        return
    for i in range(n):
        start = int(len(lines1) / n) * i
        end = int(len(lines1) / n) * (i + 1)
        if i == n - 1:
            end = len(lines1)
        l1 = lines1[start:end]
        l2 = lines2[start:end]
        with open("part" + str(i) + "updated.txt", "w") as f:
            lines = []
            for j in range(len(l1)):
                lines.append(l1[j].rstrip("\n") + "\t|\t" + l2[j])
            f.writelines(lines)


if __name__ == "__main__":
    # split_file("ru_standart_incorrect", "eng_standart_incorrect", 1)
    split("eng_standart_incorrect", "eng_mask_incorrect", "eng_partial_incorrect")
    # combine("eng_mask_incorrect", "ru_partial_incorrect", "ru_standart_incorrect")
