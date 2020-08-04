import pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve,scorer
from urllib import request, parse
import urllib.request
import json


def get_prediction(description, company_profile, benefits):
    body = {'description': description,
                            'company_profile': company_profile,
                            'benefits': benefits}

    myurl = "http://localhost:5000/predict"
    req = urllib.request.Request(myurl)
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    jsondata = json.dumps(body)
    jsondataasbytes = jsondata.encode('utf-8')   # needs to be bytes
    req.add_header('Content-Length', len(jsondataasbytes))
    #print (jsondataasbytes)
    response = urllib.request.urlopen(req, jsondataasbytes)
    return json.loads(response.read())['predictions']

if __name__=='__main__':
    description = '''Stylect is a dynamic startup that helps helps women discover and buy shoes. We’re a small team based in London that has previously worked at Google, Techstars, Pixelmator and Rocket Internet.We place a high premium on simplicity no matter what we’re working on (i.e. design, programming, marketing). We’re also a team that ships fast. We built version 1 of our app in a week, the next release (built in a month) was featured in the Apple Appstore Italy as a best new fashion app. Fast release cycles are challenging, but also very fun - which is why we love them.\xa0As we’ve grown, the projects that we’re working on have grown both in scale and in technical complexity. \xa0Stylect is looking for someone who can help us improve our backend which gathers product data; analyses/categorizes it; and shows it to thousands of users daily. Each step in the process has unique challenges that demands a strong technical background.\xa0'''
    company_profile = '''None'''
    benefits = '''We are negotiable on salary and there is the potential for equity for the right candidate.'''
    preds = get_prediction(description, company_profile, benefits)
    print(preds)
