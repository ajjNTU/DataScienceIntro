import pandas as pd

b = pd.read_csv('bankruptcy_data_set.csv')


print("Number of instances",b.shape[0])
print("Number of features", b.shape[1])
print("Columns names", b.columns.tolist())

b_no_company = b.drop('Company', axis=1)

print(b_no_company.head())

b_no_company_no_nan = b_no_company.dropna()

print("Number of instances",b_no_company_no_nan.shape[0])
print("Number of features", b_no_company_no_nan.shape[1])
print("Columns names", b_no_company_no_nan.columns.tolist())

b_no_company_no_nan.loc[:, 'WC/T_+_RE/TA'] = b_no_company_no_nan['WC/TA'] + b_no_company_no_nan['RE/TA']
b_no_company_no_nan.loc[:, 'EBIT/TA_+_S/TA'] = b_no_company_no_nan['EBIT/TA'] + b_no_company_no_nan['S/TA']

print(b_no_company_no_nan)

b2 = pd.read_csv('bankruptcy_data_set.csv')


merged_b = pd.merge(b_no_company_no_nan, b2['Company'], left_index=True, right_index=True, how='left')

print(merged_b)

ten_to_twenty_rows = merged_b.iloc[10:20]
print("10 to 20 print:")
print(ten_to_twenty_rows)
one_to_four_columns = merged_b.iloc[:, [0,1,2,3]]
print("first four columns print:")
print(one_to_four_columns)
WCTA_and_EBITTA = merged_b.loc[:, ['WC/TA', 'EBIT/TA']]
print("WCTA and EBITTA print:")
print(WCTA_and_EBITTA)
RETA_less_than_minus_20 = merged_b[merged_b['RE/TA'] < -20]
print("RETA <20 print:")
print(RETA_less_than_minus_20)
RETA_less_than_minus_20_b_is_0 = merged_b[(merged_b['RE/TA'] < -20) & (merged_b['Bankrupt'] == 0)]
print("RETA <20 and b = 0 print:")
print(RETA_less_than_minus_20_b_is_0)
RETA_less_than_minus_20_b_is_0_remove = merged_b[(merged_b['RE/TA'] < -20) & (merged_b['Bankrupt'] == 0)][['RE/TA', 'Bankrupt']]
print("RETA <20 +b = 0, less columns print:")
print(RETA_less_than_minus_20_b_is_0_remove)