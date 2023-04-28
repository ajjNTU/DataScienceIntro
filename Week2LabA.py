from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules
import pandas as pd
import random

with open('store_data.csv') as f:
    records = []
    for line in f:
        records.append(line.strip().split(','))

encoder = TransactionEncoder()
encoder_array = encoder.fit(records).transform(records)

df = pd.DataFrame(encoder_array, columns=encoder.columns_)

pd.set_option('display.width', 800)
pd.set_option('display.max_colwidth', None)

freq_items = apriori(df,min_support=0.005, use_colnames=True)
freq_items = freq_items.sort_values(by='support', ascending=False)
print(freq_items)

items_available = freq_items[freq_items['itemsets'].apply(lambda x: len(x)==1)]
print(items_available)

rules = association_rules(freq_items, metric='lift', min_threshold=1.2)
rules = rules[rules['confidence']>0.5]
rules = rules.sort_values(by='lift', ascending=False)
print("Rules All Set")
print(rules[['antecedents','consequents','confidence','lift']])


random.shuffle(records)
first_set = records[:(len(records)//2)]
first_set_en = encoder.fit(first_set).transform(first_set)
df1 = pd.DataFrame(first_set_en, columns=encoder.columns_)
second_set = records[len(records)//2:]
second_set_en = encoder.fit(second_set).transform(second_set)
df2 = pd.DataFrame(second_set_en, columns=encoder.columns_)


freq_items_1 = apriori(df1,min_support=0.005, use_colnames=True)
# freq_items_1 = freq_items.sort_values(by='support', ascending=False)
rules_1 = association_rules(freq_items_1, metric='lift', min_threshold=1.2)
rules_1 = rules_1[rules_1['confidence']>0.5]
rules_1 = rules_1.sort_values(by='lift', ascending=False)
print("Rules Set 1")
print(rules_1[['antecedents','consequents','confidence','lift']].head())

freq_items_2 = apriori(df2,min_support=0.005, use_colnames=True)
# freq_items_2 = freq_items.sort_values(by='support', ascending=False)
rules_2 = association_rules(freq_items_2, metric='lift', min_threshold=1.2)
rules_2 = rules_2[rules_2['confidence']>0.5]
rules_2 = rules_2.sort_values(by='lift', ascending=False)
print("Rules Set 2")
print(rules_2[['antecedents','consequents','confidence','lift']].head())
