import nltk
from nltk.corpus import stopwords
import re
import string
from SentTokenizer import segment
from nltk.stem import WordNetLemmatizer
from dateutil.parser import parse
from dateparser.search import search_dates
from dateutil.relativedelta import relativedelta
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

# parsing CTIME reports
def parse_impression(full_report):
    '''
    Return the impression given the full text of the report
    or empty string if impression fails to parse
    Args:
        full_report : string representing the full report text
    Returns:
        string denoting the impression parsed from the full report.
        all words are converted to lower case
    '''
    impression_words = []
    #all_words = re.findall(r"[w']+", full_report)
    all_lines = full_report.split('.')
    start = False
    for index in range(len(all_lines)):
        line = all_lines[index].lower().strip()
        if len(line) == 0:
            continue
        if ('impression:' in line or 'findings:' in line or '1.' in line):
            start = True
        if start and ('report release date' in line     or 'i have reviewed' in line     or 'electronically reviewed by' in line    or 'electronically signed by' in line    or 'attending md' in line      or 'electronic signature by:' in line):
            break
        if start or 'mass effect' in line or 'midline shift' in line or 'hemorr' in line or 'hematoma' in line or 'hernia' in line or 'herniation' in line or 'sah' in line or 'subarachnoid' in line:
            impression_words.append(line +'.')
    # Check if parsing failed
    if len([word for line in impression_words for word in line.split()]) < 2:
        return ''
    else:
        return '\n'.join(impression_words)


### processes clinical note text
def process(text):
    # Remove excess whitespace
    spaces = "                         "
    for i in range(22):
        text = text.replace(spaces[i:]," ")
    newlines = "\n\n\n\n\n\n\n\n\n\n"
    for i in range(10):
        text = text.replace(newlines[i:]," ")

    # Remove dates/propernouns for MIMIC-3
    start = text.find("[**")
    while start != -1:
        end = text.find("**]", start)
        if((text[start+3:start+7]).isnumeric()): # does it begin w/ a year?
            text = text.replace(text[start:end+3],"date") # remove dates
        else:
            text = text.replace(text[start:end+3],"propernoun") # replace w/ propernoun
        start = text.find("[**")

    # Remove periods and commans (other punctuation is preserved for CLEVER mapping)
    text = re.sub('.', ' ', text) 
    text = re.sub(',', ' ', text) 

    # CLEVER Dictionary Mapping
    cleverDict = open("data/CLEVER.txt","r")
    cleverTerms = []
    for i in range(1368):
        cleverTerms.append(cleverDict.readline()[:-1])
    cleverDict.close()
    for i in reversed(range(1368)):
        term = cleverTerms[i].split(sep = "|")
        text = text.replace(term[1], term[2]) 

    text = sent_tokenize(text)
    exclude_list = stopwords.words('english') # list of stop words
    lemmatizer = WordNetLemmatizer() # word lemmatizer

    for i in range(len(text)):
        text[i] = re.sub('['+ string.punctuation + ']', ' ', text[i]) # remove all remaining punctuation

        # eliminate propernouns
        tagged_sent = pos_tag(text[i].split()) # part-of-speech tagging
        ppns = [word for word,pos in tagged_sent if pos == 'NNP'] # find all propernouns
        for ppn in ppns:
            text[i] = text[i].replace(ppn, "") # eliminate them

        text[i] = text[i].lower() # convert all text to lowercase

        words = text[i].split(' ')
        sent = []
        for j in range(len(words)):
            if (words[j] not in exclude_list):
                # convert num to words
                if (word.isnumeric()):
                    for w in int2word(int(word)).split(): 
                        sent.append(w)

                # add all non-numeric words to the sentence
                else: sent.append(lemmatizer.lemmatize(words[j])) 

        text[i] = " ".join(sent) # add sentence back to the text variable
    return text



### Returns difference between note date and event date in note
def diffDates(d2, d1):
    diffDays = abs((d2 - d1).days)
    if diffDays <= 30: return (str(diffDays) + " day ago")
    elif (diffDays <= 364): return (str(relativedelta(d2,d1).months) + " month ago")
    return (str(relativedelta(d2,d1).years) + " year ago")


# Converts integer n to a word
def int2word(n):
    """
    convert an integer number n into a string of english words
    """
    # break the number into groups of 3 digits using slicing
    # each group representing hundred, thousand, million, billion, ...
    n3 = []
    r1 = ""
    # create numeric string
    ns = str(n)
    for k in range(3, 33, 3):
        r = ns[-k:]
        q = len(ns) - k
        # break if end of ns has been reached
        if q < -2:
            break
        else:
            if  q >= 0:
                n3.append(int(r[:3]))
            elif q >= -1:
                n3.append(int(r[:2]))
            elif q >= -2:
                n3.append(int(r[:1]))
        r1 = r

    #print n3  # test

    # break each group of 3 digits into
    # ones, tens/twenties, hundreds
    # and form a string
    nw = ""
    for i, x in enumerate(n3):
        b1 = x % 10
        b2 = (x % 100)//10
        b3 = (x % 1000)//100
        #print b1, b2, b3  # test
        if x == 0:
            continue  # skip
        else:
            t = thousands[i]
        if b2 == 0:
            nw = ones[b1] + t + nw
        elif b2 == 1:
            nw = tens[b1] + t + nw
        elif b2 > 1:
            nw = twenties[b2] + ones[b1] + t + nw
        if b3 > 0:
            nw = ones[b3] + "hundred " + nw
    return nw


### globals
ones = ["", "one ","two ","three ","four ", "five ", "six ","seven ","eight ","nine "]
tens = ["ten ","eleven ","twelve ","thirteen ", "fourteen ", "fifteen ","sixteen ","seventeen ","eighteen ","nineteen "]
twenties = ["","","twenty ","thirty ","forty ", "fifty ","sixty ","seventy ","eighty ","ninety "]
thousands = ["","thousand ","million ", "billion ", "trillion ", "quadrillion ", "quintillion ", "sextillion ", "septillion ","octillion ",
    "nonillion ", "decillion ", "undecillion ", "duodecillion ", "tredecillion ", "quattuordecillion ", "quindecillion", "sexdecillion ",
    "septendecillion ", "octodecillion ", "novemdecillion ", "vigintillion "]
