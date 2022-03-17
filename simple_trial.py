import gzip
#nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from collections import defaultdict
import math
from sklearn import linear_model

def parse(f):
    for l in gzip.open(f):
        yield eval(l)
data_review = list(parse("australian_user_reviews.json.gz"))
data_item = list(parse("australian_users_items.json.gz"))
data_game = list(parse("steam_games.json.gz"))

game_ids = {}
for j in range(len(data_game)):
    i = data_game[j]
    temp_price = "NA"
    temp_discount = "NA"
    temp_id = "NA"
    if ("id" not in i.keys()) or ("price" not in i.keys()):
        continue
    temp_id = i["id"]
    temp_price = i["price"]
    if type(temp_price)==str: temp_price = 0
    if ('discount_price' in i.keys()):
        temp_discount = i["discount_price"]
    game_ids[temp_id] = {"price": temp_price, "discount": temp_discount}

review_text = []
for i in range(len(data_review)):
    temp_review = data_review[i]["reviews"]
    for rev in temp_review:
        if rev["item_id"] in game_ids.keys():
            review_text.append(rev)
        else:
            continue

# training/validation set
# NO TIME FOR PREDICTION DARRRRRRN IT
review_text_train = review_text[:40000]
review_text_valid = review_text[40000:]

stop = set(stopwords.words('english'))
sp = set(string.punctuation)
wordCount = defaultdict(int)
for d in review_text_train:
    r = ''.join([c for c in d['review'].lower() if not c in sp])

    tokens = r.split()
    tokens = [w for w in tokens if not w in stop]
    for w in tokens:
        wordCount[w] += 1

counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()

def tfidfBuilder(data, dSize, price = True):
    #display(review_text_train.index(data))

    df = wordCount
    tf = defaultdict(int)
    words = [x[1] for x in counts[:dSize]]
    punctuation = set(string.punctuation)
    stop = set(stopwords.words('english'))

    r = ''.join([c for c in data['review'].lower() if not c in punctuation]).split()
    r = [w for w in r if not w in stop]
    for w in r:
        # Note = rather than +=, different versions of tf could be used instead
        tf[w] += 1/len(r)

    #tfidf = dict(zip(words,[tf[w] * math.log2(len(dataTrain) / df[w]) for w in words]))
    tfidfQuery = [tf[w] * math.log2(len(review_text_train) / (df[w]+1)) for w in words]
    tfidfQuery.append(1)

    if price:
        temp_id = data['item_id']
        two_price = game_ids[temp_id]

    return (tfidfQuery, two_price)

wSize = 1000
words = [x[1] for x in counts[:wSize]]

trainXY = [tfidfBuilder(d, wSize) for d in review_text_train]
xtrain = [d[0] for d in trainXY]
ytrain_price = [d[1]["price"] for d in trainXY]
ytrain_disco = [d[1]["discount"] for d in trainXY]

validXY = [tfidfBuilder(d, wSize) for d in review_text_valid]
xvalid = [d[0] for d in validXY]
yvalid_price = [d[1]["price"] for d in validXY]
yvalid_disco = [d[1]["discount"] for d in validXY]

mod = linear_model.LinearRegression()
mod.fit(xtrain, ytrain_price)

pred_train = mod.predict(xtrain)
pred_valid = mod.predict(xvalid)
svd_train = sum([x**2 for x in (ytrain_price - pred_train)]) / len(ytrain_price)
svd_valid = sum([x**2 for x in (yvalid_price - pred_valid)]) / len(yvalid_price)

words_coef = [(words[i], mod.coef_[i]) for i in range(len(words))]
display(sorted(words_coef, key=lambda x:(x[1]), reverse = True)[:50])
