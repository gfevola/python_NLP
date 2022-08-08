# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 12:44:13 2021

@author: 16317
"""

import pandas as pd
import numpy as np


jd = pd.read_excel("C:\\Users\\16317\\Documents\\Datasets\\NYCJobs.xlsx")
jd = jd.loc[0:300,:]
jdunique = jd[['Job.Description','Minimum.Qual.Requirements','Job.Category','Business.Title']].drop_duplicates().reset_index()
jdunique['index'] = range(len(jdunique))


##--run functions --
#formatted = jdunique['Minimum.Qual.Requirements'].apply(lambda x: removepunctuation(str(x)))
formatted = jdunique['Job.Description'].apply(lambda x: removepunctuation(str(x)))
jdgrams_list = formatted.apply(lambda x: splitintograms(x))

[alltagslist, allgramslist] = assemblejdtags(jdgrams_list)
[datacharacteristics, uniques, xvals] = taglist_to_grams(alltagslist,allgramslist)
[uniqueStrings, dictterms, diffmelt_mg, corpus_topics] = lda_grammodeling(uniques)


uniques_topics = pd.concat([uniqueStrings,pd.Series(corpus_topics)],axis=1)
uniques_topics = uniques_topics[['JDString','Topic',0]]

[matches_check,matches_counts] = TextsOrganizeandCompare(diffmelt_mg, allgramslist)

Writer= pd.ExcelWriter("C:\\Projects\\authsystem_new\\backend\\media\\JDs_1000.xlsx",engine='xlsxwriter')
matches_check.to_excel(Writer)
Writer.save()


#---------------------------------------
#categories--------------
#summarize xvals by entry (JD)
datacharacteristics = datacharacteristics.reset_index(drop=True)
xvals = xvals.reset_index(drop=True)
dataX = pd.concat([datacharacteristics,pd.DataFrame(xvals)],axis=1)
dataXsum = dataX.groupby('EntryNo').sum()
xvalssum = dataXsum.loc[:,1:len(xvals.columns)]
foundEntries = dataX['EntryNo'].unique()

#create tokenized from categories
jdunique['Category'] = [" ".join([stringna(jdunique['Job.Category'][i]),stringna(jdunique['Business.Title'][i])]) for i in range(len(jdunique))]
[textdf, cat_xvals,cat_tokenizer] = Texts_to_Seqs(jdunique['Category'])
cat_xvals = cat_xvals.reset_index(drop=True)
xcat_tsne = pd.DataFrame(TSNE(2).fit_transform(cat_xvals)).reset_index()

textdf1 = pd.concat([textdf,xcat_tsne],axis=1)

Writer= pd.ExcelWriter("C:\\Projects\\authsystem_new\\backend\\media\\JDsCats100 3.xlsx",engine='xlsxwriter')
textdf1.to_excel(Writer)
Writer.save()


xvaljd = xvalssum
yvaljd = cat_xvals.loc[cat_xvals.index.isin(foundEntries),:]



#_---------------------------------
# #--keras jd model
import keras as keras
##need a better model - use pretrained embeddings?
modeljd = keras.Sequential()
modeljd.add(keras.layers.Dense(len(xvaljd.columns), activation="relu"))
modeljd.add(keras.layers.Dense(len(yvaljd.columns), activation="sigmoid"))
modeljd.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
modeljd.fit(xvaljd,yvaljd, batch_size=100, epochs=20)

ytest = modeljd.predict(xvaljd)



#------------------------------

