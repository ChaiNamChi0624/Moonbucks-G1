from distribution_center import DistributionCenter
from sentiment_analysis import sentimentAnalysis
from util import load_dataset, get_country_dict
from collections import OrderedDict
import time


def get_all_info(country_codes):
    total_elapsed_time = 0

    info_dict = dict.fromkeys(country_codes, None)

    # the sample_size for distribution center should be the minimum number of stores among all the countries
    df = load_dataset()
    country_dict = get_country_dict(df)
    all_countries_n_stores = [country_dict[country_code]["n_stores"] for country_code in country_codes]
    sample_size = min(all_countries_n_stores)

    for country_code in country_codes:
        start_t = time.time()
        print(f"[{country_code}] Solving distribution center ", end="")
        
        cur_distrib_center = DistributionCenter(country_code, sample_size=sample_size)
        cur_distrib_center.solve(method="nearest-neighbour")
        cur_cost = cur_distrib_center.info()["cost"]
        
        end_t = time.time()
        elapsed_time = end_t - start_t
        total_elapsed_time += elapsed_time
        print(f"took {round(elapsed_time, 2)} s")
        
        start_t = time.time()
        print(f"[{country_code}] Sentiment analysis ", end="")
        
        cur_sentiment = sentimentAnalysis(country_code, method="freq")
        sent = cur_sentiment["score"]

        end_t = time.time()
        elapsed_time = end_t - start_t
        total_elapsed_time += elapsed_time
        print(f"took {round(elapsed_time, 2)} s")

        info_dict[country_code] = {"dist": cur_cost, "sent": sent}

    print(f"Total elapsed time : {round(total_elapsed_time, 2)} s")
    return info_dict

def score(sent, dist, total_dist, weight_sent=0.75, weight_dist=0.25):
    # weight_sent and weight_dist must sum up to 1
    if weight_dist + weight_sent != 1:
        raise Exception("weight_sent and weight_dist must sum up to 1")

    # 0.75 * sent + 0.25 * (1 - (dis / total_dist))
    # sent (higher better)
    # dis (lower better), 1 - (dis / total_dist) (higher better)
    # score (higher better)
    score = weight_sent * sent + weight_dist * (1 - (dist / total_dist))
    
    return score

def compute_all_scores(all_info, country_codes):
    dlist = all_info.copy()
    
    total_dist = 0
    for country_code in all_info.keys():
        total_dist += all_info[country_code]["dist"]
    
    for country_code in country_codes:
        dlist[country_code]["result"] = score(
            sent=all_info[country_code]["sent"], dist=all_info[country_code]["dist"], 
            total_dist=total_dist, weight_sent=0.75, weight_dist=0.25
        )
    
    sortd = OrderedDict(sorted(dlist.items(), key=lambda x: x[1]['result']))
    return sortd