import gzip, json

with gzip.open("reviews_Movies_and_TV_5.json.gz", 'rt') as f, open("reviews.csv", "w") as out:
    for line in f:
        d = json.loads(line)
        asin = d.get("asin","0")
        rating = d.get("overall",0)
        reviewer = d.get("reviewerID","0")
        text = d.get("reviewText","").replace(",", " ")
        out.write(f"{asin},{rating},{reviewer},{text}\n")
