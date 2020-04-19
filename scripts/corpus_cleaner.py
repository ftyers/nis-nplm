import re


class BasicCleaner:
    def __init__(self, filepath):
        self.filepath = filepath
        self.lines = None

    def read(self):
        with open(self.filepath) as f:
            self.lines = f.readlines()

    def trim_text(self, begin=0, end=0):
        if end == 0:
            end = len(self.lines)
        self.lines = self.lines[begin:end]

    def clean_lines(self):
        new_lines = []
        for i in range(len(self.lines)):
            self.lines[i] = re.sub(" \n$", "\n", self.lines[i])
            if re.fullmatch("^.*[.!?:»)]+\n$", self.lines[i]):
                new_lines.append(self.lines[i])
            else:
                try:
                    if re.match("¶", self.lines[i + 1]) and re.match("¶", self.lines[i + 3]):
                        new_lines.append(self.lines[i][:len(self.lines[i]) - 1] + " " + self.lines[i + 4])
                        i += 4
                except IndexError:
                    pass
        self.lines = new_lines

    def fix_direct_speech_signs(self):
        for i in range(len(self.lines)):
            reg = "^[-_–—‒⸺]\\s+"
            if re.match(reg, self.lines[i]):
                self.lines[i] = re.sub(reg, "", self.lines[i])

    def find_quoted(self, sentence):
        res = []
        new_s = ""
        balance = 0
        i = 0
        while i < len(sentence):
            s = sentence[i]
            if s == "«":
                balance += 1
            if s == "\n":
                break
            new_s += s
            if s == "»":
                balance -= 1
            if s in ".!?":
                if balance == 0:
                    i += 1
                    while i < len(sentence):
                        if sentence[i] in ".!?:)":
                            new_s += sentence[i]
                            i += 1
                        else:
                            break
                    res.append(new_s.lstrip() + "\n")
                    new_s = ""
            i += 1
        if new_s != "":
            for i in range(balance):
                new_s += "»"
            res.append(new_s.lstrip() + "\n")
        return res

    def convert_quotes(self):
        for i in range(len(self.lines)):
            l = self.lines[i]
            opened = False
            newl = ""
            for s in range(len(l)):
                if l[s] == "«":
                    symbol = "«"
                    opened = True
                elif l[s] == "»":
                    symbol = "»"
                    opened = False
                elif l[s] == "\"" or l[s] == "„":
                    if opened:
                        symbol = "»"
                        opened = False
                    else:
                        symbol = "«"
                        opened = True
                elif l[s] == "\n":
                    if opened:
                        symbol = "»\n"
                    else:
                        symbol = "\n"
                else:
                    symbol = l[s]
                newl += symbol
            self.lines[i] = newl

    def split_sentence(self, sentence):
        new_lines = []
        parts = re.split("[.!?]+", sentence)
        curr_delim_index = 0
        for i in range(len(parts)):
            p = parts[i]
            curr_delim_index += len(p)
            if p != "\n" and p != "" and not re.match("^[,!.?]", p):
                if "\n" not in p:
                    try:
                        missed = sentence[curr_delim_index]
                        p += missed + "\n"
                    except IndexError:
                        pass
                new_lines.append(p.lstrip())
            curr_delim_index += 1
        return new_lines

    def into_sentences(self):
        new_lines = []
        for l in self.lines:
            quoted = self.find_quoted(l)
            for q in quoted:
                if "«" in q:
                    if len(re.findall("«", q)) == 1:
                        parts = re.findall("(:\s+«.*»,\s?-\s?)|(:\s+«.*»)|(«.*»,\s?-\s?)", q)
                        if len(parts) == 1:
                            p = parts[0]
                            for item in p:
                                if item != "":
                                    p = item
                                    break
                            q = q.split(p)
                            if not re.fullmatch("^.*[.!?:»)]+$", q[0]):
                                q[0] += "."
                            splits = self.split_sentence(q[0])
                            for s in splits:
                                new_lines.append(s)
                            p = re.sub("(:\s+)|(»,\s?-\s?)|[«»]", "", p)
                            if not re.fullmatch("^.*[.!?:»)]+$", p):
                                p += "."
                            splits = self.split_sentence(p)
                            for s in splits:
                                new_lines.append(s)
                            splits = self.split_sentence(q[1])
                            for s in splits:
                                new_lines.append(s)
                            continue
                        q = re.sub("[«»]", "", q)
                        splits = self.split_sentence(q)
                        for s in splits:
                            new_lines.append(s)
                else:
                    splits = self.split_sentence(q)
                    for s in splits:
                        new_lines.append(s)
        self.lines = new_lines

    def write(self, filepath):
        to_write = []
        for l in self.lines:
            if not re.match("^[,!.?—]", l):
                to_write.append(l)
        with open(filepath, "w") as f:
            f.writelines(to_write)


class FictionCleaner(BasicCleaner):
    def clean_lines(self):
        new_lines = []
        for i in range(len(self.lines)):
            if self.lines[i] == "* * *":
                continue
            self.lines[i] = re.sub("^\t", "", self.lines[i])
            self.lines[i] = re.sub(" \n$", "\n", self.lines[i])
            if re.fullmatch("^.*[.!?:»)]+\n$", self.lines[i]):
                new_lines.append(self.lines[i])
            else:
                try:
                    if re.match("^\n$", self.lines[i + 1]) and re.match("^\n$", self.lines[i + 3]):
                        new_lines.append(self.lines[i][:len(self.lines[i]) - 1] + " " + self.lines[i + 4])
                        i += 4
                except IndexError:
                    pass
        self.lines = new_lines

    def write(self, filepath):
        to_write = []
        for l in self.lines:
            if not re.match("^[,!.?—]", l) and len(re.findall("[\d*]", l)) == 0 and l != "C.":
                to_write.append(re.sub("(^|\n)—\s+", "\n", re.sub("\t+", "\n", l)))
        with open(filepath, "w") as f:
            f.writelines(to_write)


class FolkloreCleaner(FictionCleaner):
    def write(self, filepath):
        to_write = []
        for l in self.lines:
            if not re.match("^[,!.?—•]", l) and len(re.findall("[\d*]", l)) == 0 and l != "C.":
                fixed = re.sub("(^|\n)—-•\s+", "\n", re.sub("\t+", "\n", l))
                to_write.append(re.sub(" +", " ", fixed))
        with open(filepath, "w") as f:
            f.writelines(to_write)


class MaterialyCleaner(FictionCleaner):
    def trim_text(self, begin=0, end=0):
        if end == 0:
            end = len(self.lines)
        self.lines = self.lines[begin:end]
        new_lines = []
        for l in self.lines:
            if not re.match("^\d.*", l) and not re.match("^с\..*", l):
                new_lines.append(l)
        self.lines = new_lines


def kym_sver():
    bc = BasicCleaner("texts/kym-sver.txt")
    bc.read()
    bc.trim_text()
    bc.fix_direct_speech_signs()
    bc.clean_lines()
    bc.convert_quotes()
    bc.into_sentences()
    bc.write("kym-sver.txt")


def fiction1():
    bc = FictionCleaner("texts/corporatxt/fiction1Orawetlan")
    bc.read()
    bc.trim_text(0, 1308)
    bc.fix_direct_speech_signs()
    bc.clean_lines()
    bc.fix_direct_speech_signs()
    bc.convert_quotes()
    bc.into_sentences()
    bc.write("fiction1Orawetlan")


def fiction2():
    bc = FictionCleaner("texts/corporatxt/fiction2rethew")
    bc.read()
    bc.trim_text(5, 769)
    bc.fix_direct_speech_signs()
    bc.clean_lines()
    bc.fix_direct_speech_signs()
    bc.convert_quotes()
    bc.into_sentences()
    bc.write("fiction2rethew")


def folklore1():
    bc = FolkloreCleaner("texts/corporatxt/folklore1olenevodov")
    bc.read()
    bc.trim_text(9, 1031)
    bc.fix_direct_speech_signs()
    bc.clean_lines()
    bc.fix_direct_speech_signs()
    bc.convert_quotes()
    bc.into_sentences()
    bc.write("folklore1olenevodov")


def tynenny():
    bc = FolkloreCleaner("texts/fiction/тынэнны.txt")
    bc.read()
    bc.trim_text(12)
    bc.fix_direct_speech_signs()
    bc.clean_lines()
    bc.fix_direct_speech_signs()
    bc.convert_quotes()
    bc.into_sentences()
    bc.write("тынэнны.txt")


def materialy():
    bc = MaterialyCleaner("texts/folklore/bogoraz materialy i mify/ChukcheeTexts-Bogoras-cyryllic.txt")
    bc.read()
    bc.trim_text(3, 3430)
    bc.fix_direct_speech_signs()
    bc.clean_lines()
    bc.fix_direct_speech_signs()
    bc.convert_quotes()
    bc.into_sentences()
    bc.write("ChukcheeTexts-Bogoras-cyryllic.txt")


def silniy():
    bc = MaterialyCleaner("texts/folklore/kto samyi silnyi na zemle.txt")
    bc.read()
    bc.trim_text(8, 1442)
    bc.fix_direct_speech_signs()
    bc.clean_lines()
    bc.fix_direct_speech_signs()
    bc.convert_quotes()
    bc.into_sentences()
    bc.write("kto samyi silnyi na zemle.txt")


def skazki_olen():
    bc = MaterialyCleaner("texts/folklore/Skazki chukchej-olenevodov.txt")
    bc.read()
    bc.trim_text(7, 1031)
    bc.fix_direct_speech_signs()
    bc.clean_lines()
    bc.fix_direct_speech_signs()
    bc.convert_quotes()
    bc.into_sentences()
    bc.write("Skazki chukchej-olenevodov.txt")


def skazki_sobr_beli():
    bc = MaterialyCleaner("texts/folklore/skazki sobrannye belikovym.txt")
    bc.read()
    bc.trim_text(3, 4391)
    bc.fix_direct_speech_signs()
    bc.clean_lines()
    bc.fix_direct_speech_signs()
    bc.convert_quotes()
    bc.into_sentences()
    bc.write("skazki sobrannye belikovym.txt")


def skazki_sobr_yatg():
    bc = MaterialyCleaner("texts/folklore/skazki sobrannye yatgyrgynom.txt")
    bc.read()
    bc.trim_text(5, 1580)
    bc.fix_direct_speech_signs()
    bc.clean_lines()
    bc.fix_direct_speech_signs()
    bc.convert_quotes()
    bc.into_sentences()
    bc.write("skazki sobrannye yatgyrgynom.txt")


def find_bad_strings(src, dest_ok, dest_to_fix):
    ok = []
    to_fix = []
    with open(src) as f:
        for l in f.readlines():
            l = re.sub('[–•≪~_»]', '', l)
            if not re.match('[^а-яА-Я.,!?()\-\n\'ёЁӄӈӃӇԒԓ:∅>ʔ ]', l):
                ok.append(l)
            else:
                to_fix.append(l)
    with open(dest_ok, 'w') as f:
        f.writelines(ok)
    with open(dest_to_fix, 'w') as f:
        f.writelines(to_fix)


def fix_apostrophes(src, dest):
    strs = []
    with open(src) as f:
        for l in f.readlines():
            new_line = ""
            delete_apost = False
            found_vowel_index = -1
            for i in range(len(l)):
                char = l[i]
                if char == " ":
                    delete_apost = False
                    found_vowel_index = -1
                    new_line += char
                    continue
                if char == "\n":
                    new_line += char
                    continue
                prev_char = ''
                if i > 0:
                    prev_char = l[i - 1]
                if found_vowel_index == -1 and char.lower() in "аоуыиэ":
                    found_vowel_index = i
                if char == "'":
                    if not delete_apost and prev_char.lower() in "аоуыиэ" and found_vowel_index == i - 1:
                        new_line += "ʔ"
                        delete_apost = True
                        continue
                    continue
                new_line += char
            strs.append(new_line)
    with open(dest, 'w') as f:
        f.writelines(strs)


if __name__ == "__main__":
    # find_bad_strings("../corpora/ru_standard_v2", "../corpora/ru_standard_v3", "../corpora/ru_standard_v3_to_fix")
    fix_apostrophes("../corpora/ru_standard_v3", "../corpora/ru_standard_v4")
