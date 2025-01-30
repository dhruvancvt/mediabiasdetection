import pandas 


prefix = 'https://'

words = ["Sponsor", "toggle"]

data = pandas.read_csv("/Users/dhruvanavinchander1/Desktop/mediabiasdetection/news_bias_dataset.csv")

c = list()
c = data.columns.tolist()
for i in range(len(c)): #Loop every column
    for word in words: #Loop for every word
        c[i] = c[i].replace(word,'')



data.to_csv("/Users/dhruvanavinchander1/Desktop/mediabiasdetection/news_bias_dataset.csv",index=False)