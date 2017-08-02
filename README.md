# KNN_Text
# Sentiment Analysis Text Classification using Bag of Words and TF-IDF models with K-Nearest Neighbor Algorithm
## Catelyn Scholl, John W. Sheppard

This project was created for the NSF REU 2017 at Montanan State University in Bozeman, Montana. This project determines a postive or negative sentiment based on the review text and utilizes either the bag of words model or tf-idf model in K Nearest Neighbor classification.


This code uses Python 3.5.3 and I used Pycharm as the IDE. Both the project interpreter default settings and the configurations must be in Python 3.5.3. Several modules may need to be imported including: bs4, nltk, numpy, pandas, requests, scikit-learn, scipy, and sklearn. To import modules in Pycharm go to file -> default settings -> project interpreter -> plus symbol and enter the name of the module.


**Txt to CSV in processing folder:**
This creates a .csv file from one of the four .txt files.

Must specify the name of the file to open in "file = open('../example_name.txt', "r", encoding='utf-8')" and must specify saved name in "with open ('../example_name.csv', 'w', newline='', encoding='utf-8') as csvfile:".


**KNN Algorithm:**
This is a classification algorithm based on the k-nearest neighbors. We decided to use the BOW model with the jaccard metric and the TFIDF model with the cosine metric.

Both BOW and TFIDF methods must specify the name of the .csv file to open in "training_data = util.get_parser_data("example_name.csv")"

Variables that can be changed in both dictionary models: remove_stopwords, lemmatize, stemmer

Variables that can be changed in KNeighborsClassifier: num_neighbors, weights, algorithm, and distance metric. Documentation on KNeighbors classifier can be found here: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html


*Please note that the dataset used in this project is an example data set. Due to certain ownership restrictions we were not able to publicize the original patent dataset. The results are similar to that of the original dataset, however. The sentiment analysis data set was taken from the UCI Machine Learning Repository and is cited in the datainformation.txt file.*

Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

Special thanks to Amy Peerlinck and Na'Shea Wiesner!



