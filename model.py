import pickle
from botnoi import cv

modFile = 'sushi.mod'
mod = pickle.load(open(modFile, 'rb'))


def predictimg(imgurl):
    print('predict image running')
    a = cv.image(imgurl)
    feat = a.getmobilenet()
    res = mod.predict([feat])
    return res

