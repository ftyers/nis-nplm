import re


def convert_segmented(src, dest):
    sents = []
    with open(src) as f:
        curr_line = ""
        i = 0
        for line in f.readlines():
            i += 1
            line = line.rstrip("\n")
            if len(line) == 0:
                if curr_line[len(curr_line) - 1] == ">":
                    curr_line = curr_line[:len(curr_line) - 1]
                sents.append(curr_line + "\n")
                curr_line = ""
                continue
            if line[0] == "#":
                continue
            parts = line.split(" ")
            if len(parts) != 2:
                print("expected to have two parts")
                return
            token, morph = parts
            if re.match(".-MORPH", token) and re.match(".-MORPH", morph):
                if len(curr_line) == 0:
                    continue
                if curr_line[len(curr_line) - 1] == ">":
                    curr_line = curr_line[:len(curr_line) - 1]
                curr_line += " "
                continue
            if morph == "S-MORPH":
                curr_line += token + ">"
            elif morph == "B-MORPH" or morph == "M-MORPH":
                curr_line += token
            elif morph == "E-MORPH":
                curr_line += token + ">"
        if curr_line != "":
            if curr_line[len(curr_line) - 1] == ">":
                curr_line = curr_line[:len(curr_line) - 1]
            sents.append(curr_line + "\n")
    with open(dest, "w") as f:
        f.writelines(sents)


if __name__ == "__main__":
    convert_segmented("../corpora/coupus.out", "../corpora/corpus_as_standart")
