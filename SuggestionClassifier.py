import tensorflow as tf
from tensorflow.contrib import learn
import pandas as pd
# import fasttext as ft
import csv
import sys
import numpy as np
import sys, re
from collections import defaultdict
import random
from random import shuffle
from nltk.tokenize import word_tokenize
import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("suggestion_data_file", "./suggestion.txt", "Data source for the suggestion data.")
tf.flags.DEFINE_string("non_suggestion_data_file", "./non_suggestion.txt", "Data source for the non suggestion data.")

# Misc Parameters
tf.flags.DEFINE_boolean("over_sampling", False, "Allow oversampling minor class")
tf.flags.DEFINE_boolean("under_sampling", True, "Allow undersampling major class")

FLAGS = tf.flags.FLAGS

pd.set_option('display.max_colwidth', -1)

# Default encoding for read csv file is utf-8 --> This prone error
# csv = pd.read_csv('Subtask-A-master/Training_Full_V1.3 .csv', encoding = "utf-8")

# Change to another encoding
# Not cp coding just iso
file_encoding_cp = 'cp1252'
file_encoding_iso = 'iso-8859-1'
file_encoding_latin = 'latin1'

# It might be an issue with the delimiters in your data and the first row
# To solve it, try specifying the sep and/or header arguments when calling read_csv. For instance,
#
# df = pandas.read_csv(fileName, sep='delimiter', header=None)
# In the code above, sep defines your delimiter and header=None tells pandas that your source data has
# no row for headers / column titles. Thus saith the docs: "If file contains no header row,
# then you should explicitly pass header=None".
# In this instance, pandas automatically creates whole-number indices for each field {0,1,2,...}.
#
# According to the docs, the delimiter thing should not be an issue.
# The docs say that "if sep is None [not specified], will try to automatically determine this."
# I however have not had good luck with this, including instances with obvious delimiters.
# training_fileA = 'Training_Full.csv'
training_fileA = 'Training_Full_V1.3.csv'
training_fileB = 'V1.4_Training.csv'

# training_fileB = "Subtask-B-master/Training_Full_V1.3 .csv"
# Only read lines without error in codec --> there are 2 lines in error
df = pd.read_csv(training_fileB, skiprows=2, encoding=file_encoding_iso,
                 error_bad_lines=False, engine='python', header=None, names=["old_index", "text", "class"]).replace('"',
                                                                                                                    '',
                                                                                                                    regex=True)


# #word_tokenize accepts a string as an input, not a file.
# stop_words = set(stopwords.words('english'))
# file1 = open("text.txt")
# line = file1.read()# Use this to read file content as a stream:
# words = line.split()
# for r in words:
#     if not r in stop_words:
#         appendFile = open('filteredtext.txt','a')
#         appendFile.write(" "+r)
#         appendFile.close()

def build_data_cv(data_folder, cv=10, clean_string=True):
    print("Data folder:",data_folder)
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_file = data_folder[0]
    # print("post file:", pos_file)
    neg_file = data_folder[1]
    vocab = defaultdict(float)
    with open(pos_file, "r", encoding="utf8") as f:
        for line in f:
            # print("line:", line)
            rev = []
            rev.append(line.strip())
            # print("rev:", rev)
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":1,
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    with open(neg_file, "r", encoding="utf8") as f:
        for line in f:
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":0,
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": np.random.randint(0,cv)}
            revs.append(datum)
    return revs, vocab

def text_preprocess(data_folder, clean_string=True):
    print("Data folder:",data_folder)
    """
    Loads data and split into 10 folds.
    """
    revs = []
    pos_file = data_folder[0]
    npos_file = data_folder[1]

    # positive_examples = list(open(pos_file, "r", encoding='utf-8').readlines())
    # positive_examples = list(open(pos_file, "r", encoding='utf-8').writelines())
    # positive_examples = [s.strip() for s in positive_examples]

    # with open(pos_file, "r", encoding="utf8") as f:
    #     with open(npos_file, 'a', encoding="utf8") as the_file:
    #         the_file.write(line+'\n')

    with open(pos_file, "r", encoding="utf8") as f:
        for line in f:
            # print("Original:", line)

            # line = line.lower()
            # line = re.sub(r'\d+', '', line)
            # replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
            # line = line.translate(replace_punctuation)
            # line = line.strip()
            with open(npos_file, 'a', encoding="utf8") as the_file:
                the_file.write(line+'\n')

    return the_file

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    # string = re.sub(r"!", " ! ", string)
    string = re.sub(r"!", " ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\(", " ", string)
    string = re.sub(r"\)", " ", string)
    # string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\?", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r'\d+', '', string)
    # string = string.replace(',', '')
    # replace_punctuation = str.maketrans(string.punctuation, ' '*len(str.punctuation))
    # string = string.translate(replace_punctuation)
    return string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data(file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    examples = list(open(file, "r", encoding='utf-8').readlines())
    examples = [s.strip() for s in examples]

    # Split by words
    x_text = examples
    x_text = [clean_str(sent) for sent in x_text]

    return x_text

def integers(a,b):
    list = []
    for i in range(a, b+1):
       list = list + [i]
    return list

def neighbor_of_rand(num, max, num_of_neighbor=0):
    list = []
    div = int(num_of_neighbor/2)
    if num-div > 0 and num+div < max:
        list = integers(num-div, num+div)
        return list
    if num-div < 0:
        list = integers(0, num+num_of_neighbor)
        return list
    else:
        list = integers(num-num_of_neighbor, max)
        return list


def oversampling(minor_class_file, major_class_file, div, mod):
    """
    Oversampling dataframe object
    """
    # print("minor_class_file before:", minor_class_file)
    minor_class_file = load_data(minor_class_file)
    # print("minor_class_file after:", minor_class_file)
    # print("minor_class_file length:", len(minor_class_file))
    major_class_file = load_data(major_class_file)
    # print("minor_class_file before:", minor_class_file)
    # print("major_class_file length:", len(major_class_file))

    # Sort list according length of elements
    minor_class_file = sorted(minor_class_file, key=len)
    # print("minor_class_file sorted:", minor_class_file)

    # Tokenize the sorted list
    tokenized_minor = [word_tokenize(i) for i in minor_class_file]
    length_tk = len(tokenized_minor)
    # print("tokenized_minor length:", length_tk)

    # Build vocabulary
    max_sentence_length = max([len(x.split(" ")) for x in minor_class_file])
    # print("max_sentence_length:", max_sentence_length)
    adding_sentences = len(major_class_file) - len(minor_class_file)
    # print("adding_sentences:", adding_sentences)

    num_of_neighbor = 5
    # adding_sentences = 10
    for i in range(adding_sentences):
        rand_num = random.randint(1, max_sentence_length)
        # print("rand_num:", rand_num)
        # Find position of tokenized_minor
        pos = int((length_tk*rand_num)/max_sentence_length)
        # print("pos:", pos)
        # Find neighbor os rand_num
        neighbors=[]
        neighbors = neighbor_of_rand(pos, length_tk, num_of_neighbor)
        # print("neighbors:", neighbors)

        # Create a list of tokens
        list_tk = []
        for k in neighbors:
            # print("key:", k)
            # print("tokenized_minor k:", tokenized_minor[k])
            list_tk = list_tk + tokenized_minor[k-1]

        # print("list_tk:", list_tk)
        test = len(tokenized_minor[neighbors[0]])
        # print("length test:", test)
        shuffle(list_tk)

        # added_sen = random.sample(list_tk, len(tokenized_minor(pos)))
        # print("shuffle list_tk:", list_tk)
        list_tk = list_tk[:test]
        # print("added_words:", list_tk)
        added_sentence = ' '.join(list_tk)
        # print("added_sentence:", added_sentence)
        minor_class_file.append(added_sentence)
    shuffle(minor_class_file)
    # print("minor_class_file after:", minor_class_file)
    return minor_class_file

def undersampling(minor_class_file, major_class_file):
    """
    Undersampling dataframe object
    """
    # print("minor_class_file before:", minor_class_file)
    minor_class_file = load_data(minor_class_file)
    # print("minor_class_file after:", minor_class_file)
    # print("minor_class_file length:", len(minor_class_file))
    major_class_file = load_data(major_class_file)
    # print("minor_class_file before:", minor_class_file)
    # print("major_class_file length:", len(major_class_file))

    shuffle(major_class_file)
    # Tokenize the sorted list
    tokenized_minor = [word_tokenize(i) for i in major_class_file]

    removing_sentences = len(major_class_file) - len(minor_class_file)
    # print("removing_sentences:", removing_sentences)
    # list_tk = []
    # list_tk = list_tk[:test]
    major_class_file = major_class_file[0:len(minor_class_file)]
    # print("major_class_file length:", len(major_class_file))
    # print("major_class_file:", major_class_file)
    return major_class_file

if __name__=="__main__":
    # Print out some first lines of dataset
    # print(df.head)
    # Delete the "index" column from the dataframe and print out shape of the dataset
    df = df.drop("old_index", axis=1)
    # print(df.head)

    # Shuffle dataframe
    # print("non_suggestion.head==========",non_suggestion.head)
    # no_frauds = len(df[df['Class'] == 1])
    # non_fraud_indices = df[df.Class == 0].index
    # random_indices = np.random.choice(non_fraud_indices,no_frauds, replace=False)
    # fraud_indices = df[df.Class == 1].index
    # under_sample_indices = np.concatenate([fraud_indices,random_indices])
    # under_sample = df.loc[under_sample_indices]
    #
    # print("df.head", df.head)
    # print("under_sample", under_sample.head)

    # print("dff", list(df))
    # df.loc[df['class'] == 0]
    df['class'] = df['class'].astype(str)
    suggestion = df.loc[df['class'] == '1']
    suggestion = suggestion.drop("class", axis=1)
    non_suggestion = df.loc[df['class'] == '0']
    non_suggestion = non_suggestion.drop("class", axis=1)

    # print("non_suggestion.head==========",non_suggestion.head)
    # min=suggestion.shape[0]
    # max=non_suggestion.shape[0]-1
    # # range=max-min
    # range=8034-6065
    # print("min", min)
    # # print(range)
    # # print(non_suggestion[0,min])
    # non_suggestion.drop(non_suggestion.index[len(non_suggestion)-range])

    # if os.path.exists("non_suggestion.txt"):
    #   os.remove("non_suggestion.txt")

    suggestion.to_csv("suggestion.txt", sep=' ', encoding='utf-8', mode='w+', header=False, index=False)
    non_suggestion.to_csv("non_suggestion.txt", sep=' ', encoding='utf-8', mode='w+', header=False, index=False)

    # Major class is non suggestion, minor class is suggestion
    num_sen_ns = non_suggestion.shape[0]
    num_sen_s  = suggestion.shape[0]

    if FLAGS.over_sampling==True:
        if num_sen_ns > num_sen_s:
            d , m = divmod(num_sen_ns,num_sen_s)
            suggestion = oversampling(FLAGS.suggestion_data_file, FLAGS.non_suggestion_data_file, d, m)
            suggestion = pd.DataFrame(np.array(suggestion).reshape(len(suggestion), 1))
            suggestion.to_csv("suggestion.txt", sep=' ', encoding='utf-8', mode='w+', header=False, index=False)
        else:
            d, m = divmod(num_sen_s,num_sen_ns)
            non_suggestion = oversampling(FLAGS.non_suggestion_data_file, FLAGS.suggestion_data_file, d, m)
            non_suggestion = pd.DataFrame(np.array(non_suggestion).reshape(len(non_suggestion), 1))
            non_suggestion.to_csv("non_suggestion.txt", sep=' ', encoding='utf-8', mode='w+', header=False, index=False)

    if FLAGS.under_sampling==True:
        if num_sen_ns > num_sen_s:
            non_suggestion= undersampling(FLAGS.suggestion_data_file, FLAGS.non_suggestion_data_file)
            non_suggestion = pd.DataFrame(np.array(non_suggestion).reshape(len(non_suggestion), 1))
            non_suggestion.to_csv("non_suggestion.txt", sep=' ', encoding='utf-8', mode='w+', header=False, index=False)
        else:
            suggestion = undersampling(FLAGS.non_suggestion_data_file, FLAGS.suggestion_data_file)
            suggestion = pd.DataFrame(np.array(suggestion).reshape(len(suggestion), 1))
            suggestion.to_csv("suggestion.txt", sep=' ', encoding='utf-8', mode='w+', header=False, index=False)
    # print(suggestion)
    # print(len(suggestion))

    # data_folder_s = ["non_suggestion.txt","nnon_suggestion.txt"]
    # suggestion=text_preprocess(data_folder_s)

