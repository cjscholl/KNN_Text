import csv
import analysis.vectorize


def txt_to_csv():
    '''Method for creating .csv file from .txt file given the filename'''

    file = open('../imdb_labelled.txt', "r", encoding='utf-8')
    li = file.readlines()
    file.close()
    header = li[0]
    li.remove(li[0])
    sentiment = []
    sentence = []
    sentencetrimmed = []

    for item in li:
        sentiment.append(item[-2])
        new = item.replace(item[-2], "")
        sentence.append(new)

    for item in sentence:
        sentencetrimmed.append(item.strip())

    for sent in sentencetrimmed:
        print(sent)
        analysis.vectorize.cleanText(sent, True, False)


    with open ('../imdb_labelled.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter = '\t', quoting=csv.QUOTE_MINIMAL)
        csvWriter.writerow(['sentence', 'sentiment'])
        for sentence, sentm in zip(sentencetrimmed, sentiment):
            csvWriter.writerow([sentence, sentm])

