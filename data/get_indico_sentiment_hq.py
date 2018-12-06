import os
os.chdir('data')
import indicoio
from pandas import read_csv, DataFrame, concat
indicoio.config.api_key = '7607a2e29588d17b0626ed8b5c0206f5'

current_indico_hq = read_csv("indico_sentiment_hq_key.csv",encoding='utf-8')
statements=read_csv("train.csv",encoding='utf-8')

sentiments_from_solver = statements.loc[:,["statement1","firstSentiment"]].rename(index=str, columns = {"statement1":"statement","firstSentiment":"sentiment_hq"})
sentiments_from_solver2 = statements.loc[:,["statement1","scondSentiment"]].rename(index=str, columns = {"statement2":"statement","secondSentiment":"sentiment_hq"})
sentiments_from_solver = concat([sentiments_from_solver, sentiments_from_solver2]).drop_duplicates(subset="statement")
current_indico_hq = concat([current_indico_hq, sentiments_from_solver]).drop_duplicates(subset="statement")



statements=list(statements.loc[:,'statement1'])+list(statements.loc[:,'statement2'])

#Note: statements are not processed. Currently the backend does not process statements either
sentiment_needed = set(list(statements))
print("all statements: ",len(sentiment_needed))
sentiment_have = set(list(current_indico_hq.loc[:,'statement'].apply(str)))
print("Currently have: ",len(sentiment_have))
sentiment_needed = sentiment_needed - sentiment_have
print("Need: ",len(sentiment_needed))
sentiment_needed = list(sentiment_needed)
# sentiment_needed = sentiment_needed[:500]
# print("To get: ",len(sentiment_needed))
# new_sentiments = indicoio.sentiment_hq(sentiment_needed)
# print(len(new_sentiments))

# to_add=DataFrame({'statement': sentiment_needed, 'sentiment_hq': new_sentiments},index=range(current_indico_hq.shape[0],current_indico_hq.shape[0]+len(sentiment_needed)))
# current_indico_hq=concat([current_indico_hq,to_add])
# current_indico_hq.to_csv("data/lookups and raw data/indico_sentiment_hq_key_(matching_data_only).csv",index=False,encoding='utf-8')

