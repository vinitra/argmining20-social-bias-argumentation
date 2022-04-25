#!/usr/bin/python3
"""
Perform a similar test as Caliskan et al (this code does not do a
permutation test for statistical significance).  You  will provide, in order

  the model name on huggingface.co/models
  the two target words (e.g, gender_m and gender_f)
  the two attribute words (e.g., pleasant and unpleasant)

For example:
./weatTest.py bert-base-multilingual-cased gender_m gender_f pleasant unpleasant

You can find the files containing the list of target/attribute words in the
 wordlists directory.  The program also provides a visualization of the
 similarities between the targets and attributes
Original code copyright (C) 2019  Ameet Soni, Swarthmore College
Email: asoni1@swarthmore.edu

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from utilities import *
import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, BertModel

datadir = "wordlists/"

def loadwordlist(filename, male_word_lists=False):
    """loads and returns all words in the given file.  Omits words not in the
    word embedding matrix"""
    if not os.path.exists(datadir+filename+".txt"):
        print("Illegal target or attribute: %s" % filename)
        print("Options are: ")
        for f in os.listdir(datadir):
            print(os.path.splitext(f)[0])
        sys.exit()
    with open(datadir+filename+".txt",'r') as f:
        words = [[], []]
        for w in f.readlines():
            toInsert = w.strip().lower().split(' ')
            words[0].append(toInsert[0])
            if male_word_lists:
                words[1].append(toInsert[0])
            else:
                words[1].append(toInsert[-1])
    return words

def getAverageSimilarity(targetVec, targetLength, attrVectors, attrLengths):
    """Calculates average similarity between words in the target list and attribute
    list"""
    # print(targetVec)
    # print(targetLength)
    # print(attrVectors[0])
    # print(attrLengths[0])
    return np.average([cosine_similarity(targetVec, targetLength, attrVectors[i], attrLengths[i]) for i in range(attrVectors.shape[0])])

def getListData(conceptWords, tokenizer, transformer):
    vectors = [[], []]
    lengths = [[], []]
    for target_idx in range(len(conceptWords[0])):
        if len(conceptWords[0][target_idx]) > 0:
            word_id = tokenizer.encode(conceptWords[0][target_idx])[1]
            v = transformer.embeddings.word_embeddings.weight[word_id].detach().numpy()
            if conceptWords[0][target_idx] == conceptWords[1][target_idx]:
                v2 = v
            else:
                word_2 = tokenizer.encode(conceptWords[1][target_idx])[1]
                v2 = transformer.embeddings.word_embeddings.weight[word_2].detach().numpy()
            vectors[0].append(v)
            lengths[0].append(np.linalg.norm(v))
            vectors[1].append(v2)
            lengths[1].append(np.linalg.norm(v2))
    vectors[0] = np.array(vectors[0])
    vectors[1] = np.array(vectors[1])
    lengths[0] = np.array(lengths[0])
    lengths[1] = np.array(lengths[1])
    return (vectors, lengths)

def rankAttributes(targetData, targetLengths, attrData, attrLengths, attrWords, n= 5):
    """Return the n highest similarity scores between the target and all attr"""
    attrSims = [getAverageSimilarity(attrData[i], attrLengths[i], targetData, targetLengths) for i in range(attrData.shape[0])]
    return attrWords[np.argsort(attrSims)[-n:]]

def run_weat(rundirect, male_word_lists=False):
    #parse inputs, load in glove vectors and wordlists
    tokenizer = AutoTokenizer.from_pretrained(rundirect[0])
    model = BertModel.from_pretrained(rundirect[0])
    target1Name = rundirect[1]
    target2Name = rundirect[2]
    attr1Name = rundirect[3]
    attr2Name = rundirect[4]

    target1 = loadwordlist(target1Name)
    target2 = loadwordlist(target2Name)
    attribute1 = loadwordlist(attr1Name, male_word_lists)
    attribute2 = loadwordlist(attr2Name, male_word_lists)

    target1Vecs, target1Lengths = getListData(target1, tokenizer, model)
    target2Vecs, target2Lengths = getListData(target2, tokenizer, model)
    attr1Data, attr1Lengths = getListData(attribute1, tokenizer, model)
    attr2Data, attr2Lengths = getListData(attribute2, tokenizer, model)

    #Find more similar attribute words for each target list
    print()
    print("Top 5 most similar attribute words to %s:" % target1Name)

    topWordsT1 = rankAttributes( target1Vecs[0], target1Lengths[0], np.concatenate([attr1Data[0], attr2Data[0]]),
            np.concatenate([attr1Lengths[0], attr2Lengths[0]]), np.concatenate([attribute1[0], attribute2[0]]))
    for word in topWordsT1[::-1]:
        print("\t"+word)

    print()
    print("Top 5 most similar attribute words to %s:" % target2Name)

    topWordsT2 = rankAttributes( target2Vecs[0], target2Lengths[0], np.concatenate([attr1Data[1], attr2Data[1]]),
            np.concatenate([attr1Lengths[1], attr2Lengths[1]]), np.concatenate([attribute1[1], attribute2[1]]))
    for word in topWordsT2[::-1]:
        print("\t"+word)

    print()
    #calculate similarities between target 1 and both attributes
    targ1attr1Sims = [getAverageSimilarity( target1Vecs[0][i], target1Lengths[0][i], attr1Data[0], attr1Lengths[0])
        for i in range( target1Vecs[0].shape[0])]
    targ1attr2Sims = [getAverageSimilarity( target1Vecs[0][i], target1Lengths[0][i], attr2Data[0], attr2Lengths[0])
        for i in range( target1Vecs[0].shape[0])]
    targ1SimDiff = np.subtract(targ1attr1Sims, targ1attr2Sims)

    #calculate similarities between target 2 and both attributes
    targ2attr1Sims = [getAverageSimilarity( target2Vecs[1][i], target2Lengths[1][i], attr1Data[1], attr1Lengths[1])
        for i in range( target2Vecs[1].shape[0])]
    targ2attr2Sims = [getAverageSimilarity( target2Vecs[1][i], target2Lengths[1][i], attr2Data[1], attr2Lengths[1])
        for i in range( target2Vecs[1].shape[0])]
    targ2SimDiff = np.subtract(targ2attr1Sims, targ2attr2Sims)

    #effect size is avg difference in similarities divided by standard dev
    d = (np.average(targ1SimDiff) - np.average(targ2SimDiff))/np.std(np.concatenate((targ1SimDiff,targ2SimDiff)))


    print()
    print("Calculating effect size.  The score is between +2.0 and -2.0.  ")
    print("Positive scores indicate that %s is more associated with %s than %s." % (target1Name, attr1Name, target2Name))
    print("Or, equivalently, %s is more associated with %s than %s." % (target2Name, attr2Name, target1Name))
    print("Negative scores have the opposite relationship.")
    print("Scores close to 0 indicate little to no effect.")
    print()
    print("Effect size: %.2f" % d)

    print()
    print("Plotting similarity scores...")

    fig = plt.figure(figsize=(16,8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_title("Similarities Scores for Target/Attribute Pairs")
    ax2.set_title("Difference Scores For Each Target")

    # Box plot of pairwise similarity scores
    df = pd.DataFrame()
    df["Similarity"] = np.concatenate([targ1attr1Sims,targ1attr2Sims,targ2attr1Sims,targ2attr2Sims])
    df["Pairs"] = [target1Name+"-"+attr1Name]*len(targ1attr1Sims)+[target1Name+"-"+attr2Name]*len(targ1attr2Sims)+[target2Name+"-"+attr1Name]*len(targ2attr1Sims) \
      +[target2Name+"-"+attr2Name]*len(targ2attr2Sims)
    df["Target"] = [target1Name]*len(targ1attr1Sims+targ1attr2Sims)+[target2Name]*len(targ2attr1Sims+targ2attr2Sims)
    df["Attribute"] = [attr1Name]*len(targ1attr1Sims)+[attr2Name]*len(targ1attr2Sims)+[attr1Name]*len(targ2attr1Sims) +[attr2Name]*len(targ2attr2Sims)
    sns.boxplot(x="Target", y="Similarity", hue="Attribute",data=df, ax=ax1)
    #Box plot of target bias in similarities
    df = pd.DataFrame()
    df["Difference"] = np.concatenate([targ1SimDiff, targ2SimDiff])
    df["Target"] = [target1Name]*len(targ1SimDiff) + [target2Name]*len(targ2SimDiff)
    ax = sns.boxplot(x="Target", y="Difference", data=df, ax=ax2)


    ticks = ax1.get_yticks()
    mx = max(abs(ticks[0]),ticks[-1])
    mx = int(mx*10+.99)/10.0
    ax1.yaxis.set_ticks(np.arange(-mx,mx+.1,.1))
    ticks = ax2.get_yticks()
    mx = max(abs(ticks[0]),ticks[-1])
    mx = int(mx*10+.99)/10.0
    ax2.yaxis.set_ticks(np.arange(-mx,mx+.1,.1))

    fig.subplots_adjust(wspace=0.5)
    fig.canvas.draw()

    labels = [item.get_text() for item in ax1.get_yticklabels()]
    labels[0] = "(less similar) " + labels[0]
    labels[-1] = "(more similar) " + labels[-1]
    ax1.set_yticklabels(labels)
    labels = [item.get_text() for item in ax2.get_yticklabels()]
    labels[0] = "(%s) " % attr2Name + labels[0]
    labels[len(labels)//2] = "(neutral) 0.0"
    labels[-1] = "(%s) " % attr1Name + labels[-1]
    ax2.set_yticklabels(labels)

    return plt


if __name__ == "__main__":
    run_weat(['dccuchile/bert-base-spanish-wwm-cased', 'gender_m', 'gender_f', 'pleasant', 'unpleasant'], male_word_lists=False)
