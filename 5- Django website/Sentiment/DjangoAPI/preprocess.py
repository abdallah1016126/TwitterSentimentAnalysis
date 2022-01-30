import re
import contractions
import unidecode
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from string import ascii_lowercase

def modifyContractions():
    contractions.add('isnt', 'is not')
    contractions.add('arent', 'are not')
    contractions.add('doesnt', 'does not')
    contractions.add('dont', 'do not')
    contractions.add('didnt', 'did not')
    contractions.add('cant', 'can not')
    contractions.add('couldnt', 'could not')
    contractions.add('hadnt', 'had not')
    contractions.add('hasnt', 'has not')
    contractions.add('havenot', 'have not')
    contractions.add('shouldnt', 'should not')
    contractions.add('wasnt', 'was not')
    contractions.add('werent', 'were not')
    contractions.add('wont', 'will not')
    contractions.add('wouldnt', 'would not')
    contractions.add('cannot', 'can not')
    contractions.add('can\'t', 'can not')
    contractions.add("can't've", "can not have")


def removeUnnecessary(doc):
    doc = unidecode.unidecode(doc)  # transliterates any unicode string into the closest possible representation in ascii text.
    doc = contractions.fix(doc)  # expands contractions
    doc = re.sub('[\t\n]', ' ', doc)  # remove newlines and tabs
    doc = re.sub(r'@[A-Za-z0-9_]+', '', doc)  # remove mentions
    doc = re.sub(r'#[A-Za-z0-9_]+', '', doc)  # remove hashtags
    doc = re.sub(r'https?://[^ ]+', '', doc)
    doc = re.sub(r'www.[^ ]+', '', doc)
    doc = re.sub('[^A-Za-z]+', ' ', doc)  # remove all characters other than alphabet
    doc = re.sub(' +', ' ', doc)  # substitute any number of space with one space only
    doc = doc.strip().lower()  # remove spaces from begining and end and lower the text
    return doc

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)



def lemmatizer(words):
    lemmatizer = WordNetLemmatizer()
    l = []
    for w in words:
        l.append(lemmatizer.lemmatize(w, get_wordnet_pos(w)))
    return l

def adjustStopWords():
    stop_words = set(nltk.corpus.stopwords.words('english'))
    exclude_words = set(("not", "no"))
    new_stop_words = stop_words.difference(exclude_words)

    # adding single characters to new_stop_words
    for c in ascii_lowercase:
        new_stop_words.add(c)
    return new_stop_words



# A function that replaces negationWords in a tokenized array with not concatenated with the next nonNegation word (bigram but conctenated)
# for example ['never', no', 'not', 'happy', 'journey'] will be ['nothappy', 'journey']
def bigramNegationWords(words, negationWords):
    l = []
    metNegation = False
    bigram = ''
    for w in words:
        if w in negationWords:
            if metNegation == False:
                bigram += 'not'
                metNegation = True
            else:
                continue
        else:
            if metNegation == True:
                bigram += w
                l.append(bigram)
                metNegation = False
                bigram = ''
            else:
                l.append(w)
    return l

def convToDict(words):
    freq= dict()
    for word in words:
        if word in freq:
            freq[word] +=1
        else:
            freq[word] = 1
    return freq

def prerpocessText(doc):
    negationWords = ['not', 'no', 'never']
    #once
    modifyContractions()
    stop_words = adjustStopWords()

    doc = removeUnnecessary(doc)

    doc = doc.split()

    doc = lemmatizer(doc)

    doc = [word for word in doc if word not in stop_words]

    doc= bigramNegationWords(doc, negationWords)

    return convToDict(doc)