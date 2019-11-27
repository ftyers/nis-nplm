1. `alphabet.py` -- create the alphabet for a given corpus.
2. `corpus2tensors` -- create a dictionary of tensors for each morpheme.

Formula:

```
repr(morpheme ∈ alph*) = 
    = oneHot(s₁, alph) ⊗ oneHot(r₁, morpheme) + ... +
    + oneHot(sₙ, alph) ⊗ oneHot(rₙ, morpheme)
Where:
    alph — alphabet for a given corpus
    alph* — the corpus
    oneHot — one hot encoding function
    ⊗ — tensor product (in this case equals to the outer product)
    s — symbol in the morphene
    r — role (index of a symbol within the morpheme)
    n — number of symbols in the morpheme
```

Example:

```python
alphabet = ['a', 'b', 'c', 'd']
morpheme = 'caab'
repr(morpheme) = \
    [0, 0, 1, 0] ⊗ [1, 0, 0, 0] \
    + [1, 0, 0, 0] ⊗ [0, 1, 0, 0] \
    + [1, 0, 0, 0] ⊗ [0, 0, 1, 0] \
    + [0, 1, 0, 0] ⊗ [0, 0, 0, 1] \
    = \
    [
        [0, 1, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
    ]
```

3. `autoencoder.py` (not 100% certain)

Formula:

```
morphemes = repr(morpheme ∈ alph*)*
repr(word ∈ morphemes*) = 
    = repr(morpheme₁) ⊗ oneHot(m₁, word) + ... +
    + repr(morphemeₙ) ⊗ oneHot(mₙ, word)
Where:
    morphemes — set of morphemes
    word — set of words consisting of morphemes
```

The result is a very sparse 3D matrix. A neural network is used to turn it into a dense vector (1D matrix).
