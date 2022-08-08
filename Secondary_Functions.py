# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 12:42:47 2022

@author: 16317
"""
import nltk as nltk
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from keras_preprocessing.text import Tokenizer

import String_Functions as strf

def assemblejdtags(jdgrams_list):
    
    # nltk.download('omw-1.4')
    # nltk.download('stopwords')
    # nltk.download('wordnet')
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('punkt')
    
    alltags = []
    allgrams = []
    a = 0
    for jdgram in jdgrams_list:
        #categorize each gram
        a += 1
        grambuilder = []
        for g in jdgram:
            cat = ""
            if len(g) > 20:
                if g.find("degree") + g.find("diploma")+ g.find("licensed")  > -2 :
                    cat = "Education"
                if g.find("experience") + g.find("years of") > -1:
                    cat = "Experience"
                grambuilder.append([g,cat])
 
        #pos tags
        b = 0
        fulltags = []
        if len(grambuilder) > 0:
            for g in grambuilder:
                b += 1
                tags = strf.parse_term(g[0])
                #tags = nltk.pos_tag(g[0].split(" "))
                tags = nltk.pos_tag(tags)
                lst = strf.filter_pos_tag(tags)
                reassembled = " ".join(lst[0])
                allgrams.append([a,b,g[0],reassembled])
                lst['Seq']=range(len(lst))
                lst['GramNo'] = b
                lst['Category'] = ''
                lst['Value'] = ''
                fulltags.append(lst)
                #degree terms
                if g[1]=="Education":
                    lst['Category'] = g[1]
                    lst['Value'] = strf.degreelevel(lst)
                if g[1]=="Experience":
                    lst['Category'] = g[1]
                    nums = lst.loc[lst.loc[:,1]=="CD",0]
                    if len(nums)>0:
                        lst['Value'] = nums.iloc[0]

            fulltags = pd.concat(fulltags)
            fulltags['EntryNo'] = a
            alltags.append(fulltags)

    allgrams = pd.DataFrame(allgrams)
    allgrams.columns = ['Entry','Gram','JDString','Terms']
    
    alltags = pd.concat(alltags)
    alltags = alltags.reset_index(drop=True)
    alltags = alltags.loc[alltags[0]!="",:]
    alltags.columns = ['Term','POS_Tag','Seq','GramNo','Category','Cat_Value','EntryNo']

    return([alltags,allgrams])


#------------------------------------------------
#tags_to_xval
def taglist_to_grams(alltagslist, allgramslist):
    
    jdtokenizer = Tokenizer()
    jdtokenizer.fit_on_texts(alltagslist['Term'])
    alltagslist['TokenNum'] = jdtokenizer.texts_to_sequences(alltagslist['Term'])
    wordcount = len(jdtokenizer.word_counts)
    
    
    xvals = []
    datacharacteristics = []
    #for each entry/gram
    for e in range(max(alltagslist['EntryNo'])):
        for f in range(max(alltagslist['GramNo'])):
            filt = alltagslist.loc[(alltagslist['EntryNo']==e) & (alltagslist['GramNo']==f),:]
            if len(filt) > 0:
                xval = strf.addcategories(filt['TokenNum'],wordcount,True)
                xvals.append(xval)
                datacharacteristics.append(filt.iloc[0,:])
    
    
    datacharacteristics = pd.concat(datacharacteristics,axis=1).transpose().reset_index()
    
    datacharacteristics = datacharacteristics.loc[:,["EntryNo",'GramNo','Category','Cat_Value']]
    datacharacteristics = datacharacteristics.merge(allgramslist,how="left",left_on=['EntryNo','GramNo'],right_on=['Entry','Gram'])
    
    xvals = pd.concat(xvals,axis=1).transpose() 
    xvals = xvals.loc[:,1:] #first column is empty
    
    #pca/tsne on xvals
    pca = PCA(n_components = 40)
    x_pca = pca.fit_transform(xvals)
    
    x_tsne = pd.DataFrame(TSNE(3).fit_transform(xvals),columns=['X','Y','Z'])
    datacharacteristics_mg = pd.concat([datacharacteristics,x_tsne],axis=1)
    
    #check nearest
    uniqueStrings = datacharacteristics_mg[['JDString','Terms','Category','X','Y','Z']].drop_duplicates()
    uniqueStrings = uniqueStrings.groupby(['JDString','Terms','Category']).mean().reset_index()
    uniqueStrings['index'] = range(len(uniqueStrings))
    
    return([datacharacteristics_mg, uniqueStrings, xvals])



import gensim
from gensim import corpora, models
import random

def lda_grammodeling(uniqueStrings):
    patterns = uniqueStrings.loc[uniqueStrings['Category']!="Education",'Terms'].apply(lambda x: strf.parse_term(x))
    dictionary = gensim.corpora.Dictionary(patterns)
    dictionary.filter_extremes(no_below=1, no_above=.05, keep_n=50000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in patterns]
    
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=90, id2word=dictionary, passes=5, workers=2)
    
    def reviewScore(lmodel,corpus,elem):
        score = max([(v,i) for i, v, in lmodel[corpus[elem]]])
        return np.asarray(score)
    
    def termScore(lmodel, elem):
        score = [max((v,i) for i, v in lmodel[[(elem,1)]])]
        return pd.DataFrame(score,columns=['TopicScore','Topic'])
    
    svals = pd.DataFrame([reviewScore(lda_model,bow_corpus,i) for i in range(len(bow_corpus))],columns=['TopicScore','Topic'])
    uniqueStrings = pd.concat([uniqueStrings,svals],axis=1)
    
    #dictionary terms
    dictterms = pd.DataFrame([dictionary[i] for i in range(len(dictionary))],columns=["Term"])
    termscores = pd.concat([termScore(lda_model,i) for i in range(len(dictionary))],axis=0).reset_index(drop=True)
    dictterms = pd.concat([dictterms,termscores],axis=1)
    
    
    #find the difference between TSNE values
    def matrixsubtract(arrycol):
        lst = np.array(arrycol)
        v1  = [pd.DataFrame(lst - lst[i][0]) for i in range(len(lst))]
        return(pd.concat(v1,axis=1))
    
    xcomp = np.round(matrixsubtract(uniqueStrings[['X']]),4)
    ycomp = np.round(matrixsubtract(uniqueStrings[['Y']]),4)
    zcomp = np.round(matrixsubtract(uniqueStrings[['Z']]),4)
    
    diffmat = abs(xcomp)+abs(ycomp)+abs(zcomp)
    diffmat.columns = range(len(diffmat.columns))
    diffmat['UniqueGramNo'] = [f'{i}_{random.randrange(16**7):x}' for i in range(len(diffmat))]
    diffmat = diffmat.reset_index()
    diff_indexing = diffmat[['index','UniqueGramNo']].drop_duplicates()
    
    diffmelt = pd.melt(diffmat,id_vars=['index',"UniqueGramNo"])
    diffmelt = diffmelt.merge(diff_indexing,how="inner",left_on="variable", right_on="index",suffixes=["From",'To'])
    diffmelt = diffmelt.drop(['variable'],axis=1)
    diffmelt = diffmelt.rename({'level_0':'indexFrom','index':'indexTo'},axis=1)
    diffmelt = diffmelt.loc[(diffmelt['value'] < 10) & (diffmelt['UniqueGramNoFrom'] > diffmelt['UniqueGramNoTo']),]

    diffmelt_mg = diffmelt.merge(uniqueStrings,how="left",left_on="indexFrom",right_on="index")
    diffmelt_mg = diffmelt_mg.merge(uniqueStrings,how="left",left_on="indexTo",right_on="index",suffixes=['To','From'])
    diffmelt_mg = diffmelt_mg.drop(['indexFrom','indexTo'],axis=1)
    
    corpus_topics = [ld for ld in lda_model[bow_corpus]]
    uniqueStrings = uniqueStrings.merge(diff_indexing,how='left',left_on='index',right_on='index')
    
    return([uniqueStrings, dictterms, diffmelt_mg, corpus_topics])



def TextsOrganizeandCompare(diffmelt_mg, allgramslist, jdunique):

    #merge on strings
    diffmelt_mg['SameTopic'] = diffmelt_mg['TopicFrom']==diffmelt_mg['TopicTo']
    
    diffmelt_select = diffmelt_mg[['UniqueGramNoFrom','UniqueGramNoTo','value','TermsFrom','TermsTo','TopicFrom','TopicTo','SameTopic']]
    diffmelt_select = diffmelt_select.loc[(diffmelt_select['TopicFrom']==diffmelt_select['TopicTo']) | (diffmelt_select['value']<9) ,:]
    
    #apply to grams list (to revert to entry numbers)
    matches_grams = diffmelt_select.merge(allgramslist,how="inner",left_on="TermsFrom",right_on="Terms")
    matches_grams = matches_grams.merge(allgramslist,how="inner",left_on="TermsTo",right_on="Terms",suffixes=['_From','_To'])
    matches_grams['valueINV'] = 1/np.sqrt(matches_grams['value']+1)
    matches_grams = matches_grams[matches_grams['Entry_From']!=matches_grams['Entry_To']] #filter out same matches
    matches_grams = matches_grams.drop(['TermsFrom','TermsTo'],axis=1)

    matches_counts = matches_grams.groupby(['Entry_From','Entry_To']).sum('valueINV').reset_index()
    
    #check matches quality
    matches_check = matches_counts.merge(jdunique,how='left',left_on='Entry_From',right_on='index')
    matches_check = matches_check.merge(jdunique,how='left',left_on='Entry_To',right_on='index',suffixes=['From','To'])
    matches_check = matches_check.loc[matches_check['valueINV']>1,:]
    matches_check = matches_check.loc[matches_check['Entry_From']>matches_check['Entry_To'],:]
    
    return([matches_grams, matches_counts, matches_check])


def Texts_to_Seqs(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    
    textdf = pd.DataFrame(texts)
    textdf.columns = ['Text']
    
    textdf['TermList'] = textdf.loc[:,'Text'].apply(lambda x: strf.parse_term(x))
    textdf['POS']  = textdf.loc[:,'TermList'].apply(lambda x: nltk.pos_tag(x))
    textdf['POSFilt'] = textdf['POS'].apply(lambda x: strf.filter_pos_tag(x))
    textdf['Seqs'] = textdf['POSFilt'].apply(lambda x: tokenizer.texts_to_sequences(x.loc[:,0]))
    textdf['SeqsUnlist'] =  textdf['Seqs'].apply(lambda x: strf.intseq_unlist(x))
    numwords = len(tokenizer.word_counts)+1
    
    xvals = [strf.addcategories(x,numwords,False).transpose() for x in textdf['SeqsUnlist']]  #changed addcategories since
    xvals = pd.concat(xvals)
    
    #filter columns(terms) > 1
    xvalsfilt = xvals.loc[:,xvals.sum()>1]
    
    return([textdf,xvalsfilt,tokenizer])