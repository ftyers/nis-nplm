# Fran's char RNN LM thing

Training:
```bash
$ python3 train.py data/toy.txt toy.cp
```

Inference:
```bash
$ python3 predict.py toy.cp 
```

This will give a lot of output, culminating in:
```bash
seq: [5, 9, 4, 2, 10, 0, 9, 10, 8, 7, 6, 1, 6, 0, 6, 3] #she is run>ning
ctx: [5, 9, 4, 2, 10, 0, 9, 10, 8, 7, 6, 1, 6, 0, 6, 3]
pred: [5, 4, 4, 2, 10, 2, 9, 10, 0, 8, 6, 1, 6, 1, 6, 3] #hhe es irn>n>ng
hits: 10
clicks: 11 ['s', 'he ', 'i', 's ', 'r', 'u', 'n>', 'n', 'i', 'n', 'g']
len: 15
```

Where,
* `seq` = input sequence (e.g. what is in the thing that the user typed)
* `ctx` = context
* `pred` = prediction from the LM
* `hits` = number of times the machine got it right
* `clicks` = number of clicks the user had to make
* `len` = length of the input
