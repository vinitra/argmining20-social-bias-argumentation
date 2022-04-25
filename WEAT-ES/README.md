# WEAT-ES

Spanish language bias measurement benchmark for word embeddings

Translated English benchmark with accomodations for grammatical gender
(e.g. doctor/doctora)

Based on Dr. Ameet Soni's WEATLab: https://github.com/ameetsoni/WEATLab

## Formatting word lists

Put each word on a new line, without spaces or newlines.
For words which have male and female forms, put a space or tab in between them,
on the same line

```
inteligente
nuevo nueva
doctor doctora
```

## Running with Transformers

You can compare against models from https://huggingface.co/models

Here we compare Multilingual BERT results - be sure to put the two targets
(gender_m and gender_f) in that order to be consistent with the order in the word lists

```
pip3 install transformers numpy
run_weat(['dccuchile/bert-base-spanish-wwm-cased', 'gender_m', 'gender_f', 'pleasant', 'unpleasant'])
```

## Alternatives / Updates

XWEAT, a more rigorous cross-lingual WEAT dataset, was published in 2019: https://github.com/anlausch/XWEAT

"Intrinsic Bias Metrics Do Not Correlate with Application Bias" identifies translation issues in XWEAT data,
and did not find a correlation between intrinsic bias (with embedding-level biases such as WEAT) and final application bias. https://arxiv.org/abs/2012.15859

## Sample notebook

https://colab.research.google.com/drive/1Yicr3qSkh0reKBEYwohygM3RAQdEirxf

## License

Open source, GPLv3 license
