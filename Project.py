import sys
import os
import csv
from collections import defaultdict
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import random
import time
import winsound

def get_top_hundred_subs(list_of_subs):
    sub_dict = defaultdict(float)
    for sub in list_of_subs:
        sub_dict[sub] += 1.0

    # will be of the form [(key, val), (key, val), ....]
    sorted_list = sorted(sub_dict.items(), key=lambda value: value[1])
    # returns the last hundred items in the list
    # create a list with only the keys
    sorted_list = sorted_list[-100:]
    Top100_withcounts = [tuple((y[0], y[1])) for y in sorted_list]
    return Top100_withcounts


def read_csv(file_name, cols_list):
    """
	reads the file and returns a list of tuples
	with required cols in the order supplied in cols_list
	"""
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        data = []
        cols_list=[5,11,17,19,20,21,1]
        for idx, row in enumerate(reader):
            needed_items = tuple(row[i] for i in cols_list)
            data.append(needed_items)

        return data


def add_headers(row, cols_list):
    """
	utility function to return a dict with
	row item as key and its index as value
	"""
    index_dict = {}
    for idx, item in enumerate(row):
        if item in cols_list:
            index_dict[item] = idx
    return index_dict


def clean_str(title):
    return unicode(title.lower().strip(), 'utf-8', errors='ignore')


def build_model(titles):                #model lo each subreddit min and max date pett ..oka post ki time chusi min kante thakuva or max kante ekvu unte deleting the subreddit before sending it to prediction
    model = defaultdict(lambda: defaultdict(int))
    nouns = ['NN', 'NNP', 'NNPS', 'NNS','JJ','JJR','JJS']
    setofsubredditsover_18=set()
    countover_18=defaultdict(int)
    for created, id, over_18, selftext, sub, title, author in titles:
        tagged_title = pos_tag(word_tokenize(clean_str(title)))
        for word, pos in tagged_title:
            if pos in nouns:
                model[sub][word] += 1
        if selftext is not None and len(selftext) > 0:
            tagged_selftext = pos_tag(word_tokenize(clean_str(selftext)))
            for word, pos in tagged_selftext:
                if pos in nouns:
                    model[sub][word] += 1
        if over_18 == 'true':
            setofsubredditsover_18.add(sub)
            countover_18[sub]+=1
    TotalNoofpostover_18=sum(countover_18.values())
    Justlist=list(setofsubredditsover_18)
    threshold=0.02*TotalNoofpostover_18
    listofsubredditsover_18=[x for x in Justlist if countover_18[x]>threshold]

    return model,listofsubredditsover_18

def prediction_subreddit(model,dictofmin,over_18,item):
    subreds=dictofmin.keys()
    delsubr=set()
    for subr in subreds:
        if item[0]<dictofmin[subr]:
            delsubr.add(subr)
        if item[2]=='true':
            if subr not in over_18:
                delsubr.add(subr)
    subreddit,possiblesubreds=predict_subreddit(model,item,list(delsubr))
    return subreddit,possiblesubreds


def predict_subreddit(model, post,delsubr):
    """
	calculate score for each title against every sub
	and return the sub with highest score
	"""
    nouns = ['NN', 'NNP', 'NNPS', 'NNS','JJ','JJR','JJS']
    high_score = 0
    high_score_sub = ''
    created,over_18inpost,idofpost, selftext, title,author = post
    tagged_title = pos_tag(word_tokenize(clean_str(title)))
    possiblesubreddits=set()
    for sub, words in model.iteritems():
        if sub not in delsubr :
            if author in authors[sub]:
                score = 0
                for word, pos in tagged_title:
                    if pos in nouns:
                        score += words[word]
                if selftext is not None and len(selftext) > 0:
                    tagged_selftext = pos_tag(word_tokenize(clean_str(selftext)))
                    for word, pos in tagged_selftext:
                        if pos in nouns:
                            score += words[word]
                for word,pos in tagged_title:
                    if word==sub:
                        score=score+1000
                if score > high_score:
                    high_score_sub = sub
                    high_score = score
                possiblesubreddits.add(sub)
    return high_score_sub,possiblesubreddits

def accuracy(model,data,validation):
    Falsenegative = defaultdict(float)
    Truepositive = defaultdict(float)
    Falsepositive = defaultdict(float)
    matches = 0.0
    modela = model[0]                               # Features :  1. Nouns in title and selftet                                       # 2. Birthdate of a subreddit
    over_18 = model[1]                                              # 3. Posts with Over_18
    F1score = defaultdict(float)
    for item in data:
        subreddit,possiblesubreds = prediction_subreddit(modela,dictofmin,over_18,item)
        if not subreddit :
            c=0
            potentialsubreddit=''
            for i in possiblesubreds:
                if top100counts[i]>c:
                    c=top100counts[i]
                    potentialsubreddit=i
            subreddit=potentialsubreddit
        if validation[item[1]] == subreddit:
            matches += 1
            Truepositive[subreddit] += 1
        else:
            Falsepositive[subreddit] += 1
            Falsenegative[validation[item[1]]] += 1
    ACCURACY=matches*100/(len(data)+0.0001)
    for subreddit in top100counts.keys():
        F1score[subreddit] = (2 * Truepositive[subreddit]) / ((2 * Truepositive[subreddit]) + Falsepositive[subreddit] + Falsenegative[subreddit] + 1.00)
    return ACCURACY,F1score


if __name__ == '__main__':
    # Assumes reddit.com is a placeholder for actual sub to be put in
    # Install nltk (http://www.nltk.org/install.html#windows)
    # Install nltk_data (http://www.nltk.org/data.html)
    # install all in nltk data in linux i had to type 'book'
    # which installed all the packages, and corpora

    training_file = sys.argv[1]
    # testing_file = 'test1.csv'
    #[tuple((subreddit[0],subreddit[1]))for subreddit in counttrain if subreddit[1]>(0.05*toppestsubvalue)]
    # list of tuples
    output = read_csv(training_file, ['created', 'id', 'over_18', 'selftext', 'subreddit', 'title','author'])
    counttrain = get_top_hundred_subs([x[4] for x in output if x[4] != 'reddit.com'])
    countt = defaultdict(int)
    now=time.time()
    global authors,dictofmin
    authors=defaultdict(set)
    dictofmin=defaultdict(long)
    global top100counts
    top100counts = dict(counttrain)
    Totalnumberofposts=sum(top100counts.values())
    percentagesofeachsubreddit=dict([tuple((key,(top100counts[key]/Totalnumberofposts))) for key in top100counts.keys()])
    output=[tuple(i) for i in output if top100counts.has_key(i[4])]
    output1=sorted(output.__iter__(),key=lambda value: value[0])
    del output
    setoftop100=set(top100counts.keys())
    print setoftop100
    setoftop=set()
    counter=0
    for created, id, over_18, selftext, sub, title, author in output1:
        authors[sub].add(author)
        if sub not in setoftop:
                dictofmin[sub]=created
                setoftop.add(sub)
    kfold=0
    test_datawithselftext=[]
    test_datawithtitlesonly=[]
    Validationfortotaltestset=defaultdict(str)
    CONFIDENCEinselftextKfold=[]
    ConfidenceinValidation=[]
    ConfidenceinTesting=[]
    ACCURACYinselftextKfold=[]
    ACCURACYintitlesKfold=[]
    ConfidenceintitlesKfold=[]
    ACCURACYofKfold=[]
    while(kfold<4):
        kfold += 1
        with open('gearsecond.csv', 'a') as f:
            f.write("In fold %d\n" %kfold)
            f.write('-----------------------------------------------------------------------------------------------\n')
        training_data = []
        F1scoreoftotalinselfTesting=defaultdict(float)
        F1scoreoftotalintitleTesting=defaultdict(float)
        F1scoreoftotalinTesting = defaultdict(float)
        F1scoreoftotalinValidation = defaultdict(float)
        test_data= []
        validationlist=[]
        listfortraining=[1,2,3,4,5,6,7,8,9]
        randomnumberfortrain=random.sample(listfortraining,4)
        randominstancesfortesting=list(set(listfortraining)-set(randomnumberfortrain))
        randomsintest=random.sample(randominstancesfortesting,3)
        randomforvalidation=list(set(randominstancesfortesting)-set(randomsintest))
        randomnumberfortrain.append(0)
        for item in output1:
            reminder=countt[item[4]]%10
            if reminder in randomnumberfortrain:                               # taking 50% of each subreddit for training
                training_data.append(item)
                countt[item[4]] += 1
            elif reminder in randomforvalidation:
                validationlist.append(item)
                countt[item[4]] += 1                                    # sending other 20% for validation
            else:                              #sending other 30% for testing
                if item[3] is not None and len(item[3])>0:
                    test_datawithselftext.append(tuple((item[0], item[1], item[2], item[3], item[5],item[6])))
                else:
                    test_datawithtitlesonly.append(tuple((item[0], item[1], item[2], item[3], item[5],item[6])))
                Validationfortotaltestset[item[1]]= item[4]
                countt[item[4]] += 1
        print 'Okay'
        model = build_model(training_data)
        with open('gearsecond.csv', 'a') as f:
            f.write("Model created\n")
        print 'Model created'
        ACCURACYinselftext,CONFIDENCEinselftext=accuracy(model,test_datawithselftext,Validationfortotaltestset)
        ACCURACYintitles, CONFIDENCEintitles = accuracy(model, test_datawithtitlesonly, Validationfortotaltestset)
        with open('gearsecond.csv', 'a') as f:
            f.write("Accuracy in Testing %f\n" % ACCURACYinselftext)
        ACCURACYinselftextKfold.append(ACCURACYinselftext)
        with open('gearsecond.csv', 'a') as f:
            f.write("Accuracy in Testing %f\n" % ACCURACYintitles)
        ACCURACYintitlesKfold.append(ACCURACYintitles)
        for SBreddit in top100counts.keys():
            proportion = percentagesofeachsubreddit[SBreddit]
            F1scoreoftotalinselfTesting[SBreddit] = proportion * CONFIDENCEinselftext[SBreddit]
        with open('gearsecond.csv', 'a') as f:
            f.write( "Confidence in Testing: %f \n" % sum(F1scoreoftotalinselfTesting.values()))
        CONFIDENCEinselftextKfold.append(sum(F1scoreoftotalinselfTesting.values()))
        for SBreddit in top100counts.keys():
            proportion = percentagesofeachsubreddit[SBreddit]
            F1scoreoftotalintitleTesting[SBreddit] = proportion * CONFIDENCEintitles[SBreddit]
        with open('gearsecond.csv', 'a') as f:
            f.write( "Confidence in Testing: %f \n" % sum(F1scoreoftotalintitleTesting.values()))
        ConfidenceintitlesKfold.append(sum(F1scoreoftotalintitleTesting.values()))
    with open('gearsecond.csv', 'a') as f:
        f.write('Overall stats: ----------------------------------------------------\n')
        f.write('Total avg of accuracy in only Titles %f\n' %(sum(ACCURACYintitlesKfold)/4))
        f.write('Total avg confidence in only titles  %f\n' %(sum(ConfidenceintitlesKfold)/4))
        f.write('Total avg of accuracy in self text: %f\n' % (sum(ACCURACYinselftextKfold) / 4))
        f.write('Total avg confidence in selftext %f\n' % (sum(CONFIDENCEinselftextKfold) / 4))
        f.write('Total avg of accuracy in validation: %f\n' % (sum(ACCURACYofKfold) / 4))
        f.write('Total avg confidence in validation%f\n' % (sum(ConfidenceinValidation) / 4))



