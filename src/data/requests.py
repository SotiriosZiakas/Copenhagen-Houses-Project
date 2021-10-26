import pandas as pd
import requests

results = []
url = "https://api.boliga.dk/api/v2/search/results"
for x in range(1,25):
    querystring = {"pageSize":"100","area":"1","sort":"daysForSale-a","page":f"{x}","includeds":"1"}


    headers = {"sec-ch-ua": "^\^Chromium^^;v=^\^94^^, ^\^Google"}

    r = requests.request("GET", url, headers=headers, params=querystring)

    data = r.json()

    for product in data['results']:
        results.append(product)

my_data = pd.json_normalize(results)
my_data.to_csv('boliga_results.csv')