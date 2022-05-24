# Dexter Dysthe
# Dr. Giroud
# B9326: Financial Econometrics, Panel Data
# 19 April 2022

import numpy as np
import pandas as pd
from regpyhdfe import Regpyhdfe
from scipy.stats.mstats import winsorize

# ** All datasets downloaded from WRDS restrict the data to between January 1, 1976 and December 31, 2006. ** #

# ---------------------------------------- Clean Compustat Annual Data ---------------------------------------- #
# From WRDS go to: Compustat-Capital IQ -> North America -> Fundamentals Annual. The variables we select are shown
# in the dataframe comp_annual below.

# Compustat provides additional columns of data, therefore keep only those annual variables necessary for our
# analysis. There are 330,837 initial rows. This is downloaded from Compustat North America annual.
comp_annual = pd.read_stata('compustat_annual.dta')[['gvkey', 'datadate', 'fyear', 'at', 'ch', 'csho', 'lt',
                                                     'sale', 'prcc_f', 'incorp', 'sic', 'state']]

# Exclude firms with either missing or negative assets or sales. There are 60,806 rows with either missing
# assets or sales, and 152 rows with negative assets or sales.
comp_annual = comp_annual[(comp_annual['at'].notna()) & (comp_annual['sale'].notna())]
comp_annual = comp_annual[(comp_annual['at'] >= 0) & (comp_annual['sale'] >= 0)]

# Drop any remaining firms having missing values. This removes 70,239 rows.
comp_annual = comp_annual.dropna()

# Remove firms with SIC codes 4900-4999. First, convert SIC codes from type object to type float. This will
# also be necessary for when we construct the industry-by-year fixed effect variable. This removes 7,423 rows,
# leaving us with 199,640 observations.
comp_annual = comp_annual.astype({'sic': 'float'})
reg_utility_firm_codes = [float(code) for code in range(4900, 5000)]
criterion = lambda row: row['sic'] not in reg_utility_firm_codes
comp_annual = comp_annual[comp_annual.apply(criterion, axis=1)]

# We do not need annual assets or sales for the remainder of our analysis, therefore we now discard these columns.
comp_annual = comp_annual[['gvkey', 'datadate', 'fyear', 'ch', 'csho', 'lt', 'prcc_f', 'incorp', 'sic', 'state']]

# Remove non-U.S states. This leaves us with 152,513 observations.
us_states = ["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA",
             "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM",
             "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA",
             "WV", "WI", "WY"]
comp_annual = comp_annual[(comp_annual.state.isin(us_states)) & (comp_annual.incorp.isin(us_states))]

# Create fixed effects columns: firm FE, state of location-by-year FE, and SIC-by-year FE. We follow Dr. Giroud's
# approach as in the B&M replication to create the latter two FEs. Use pd.factorize() to emulate STATA's encode
# command.
labels_gvkey, levels_gvkey = pd.factorize(comp_annual['gvkey'])
labels_state, levels_state = pd.factorize(comp_annual['state'])
labels_incorp, levels_incorp = pd.factorize(comp_annual['incorp'])
comp_annual['gvkeyn'] = labels_gvkey
comp_annual['staten'] = labels_state
comp_annual['incorpn'] = labels_incorp
# Following Dr. Giroud's approach from B&M replication. Use the same strategy for industry-by-year FE as Dr. Giroud
# used for state-by-year FE.
comp_annual['styear'] = 10000*comp_annual['staten'] + comp_annual['fyear']
comp_annual['sicyear'] = 10000*comp_annual['sic'] + comp_annual['fyear']

# Create Business Combination law dummy
comp_annual['BC'] = np.where(((comp_annual['incorp'] == 'AZ') & (comp_annual['fyear'] >= 1987)) |
                             ((comp_annual['incorp'] == 'CT') & (comp_annual['fyear'] >= 1989)) |
                             ((comp_annual['incorp'] == 'DE') & (comp_annual['fyear'] >= 1988)) |
                             ((comp_annual['incorp'] == 'GA') & (comp_annual['fyear'] >= 1988)) |
                             ((comp_annual['incorp'] == 'ID') & (comp_annual['fyear'] >= 1988)) |
                             ((comp_annual['incorp'] == 'IL') & (comp_annual['fyear'] >= 1989)) |
                             ((comp_annual['incorp'] == 'IN') & (comp_annual['fyear'] >= 1986)) |
                             ((comp_annual['incorp'] == 'KS') & (comp_annual['fyear'] >= 1989)) |
                             ((comp_annual['incorp'] == 'KY') & (comp_annual['fyear'] >= 1987)) |
                             ((comp_annual['incorp'] == 'ME') & (comp_annual['fyear'] >= 1988)) |
                             ((comp_annual['incorp'] == 'MD') & (comp_annual['fyear'] >= 1989)) |
                             ((comp_annual['incorp'] == 'MA') & (comp_annual['fyear'] >= 1989)) |
                             ((comp_annual['incorp'] == 'MI') & (comp_annual['fyear'] >= 1989)) |
                             ((comp_annual['incorp'] == 'MN') & (comp_annual['fyear'] >= 1987)) |
                             ((comp_annual['incorp'] == 'MO') & (comp_annual['fyear'] >= 1986)) |
                             ((comp_annual['incorp'] == 'NE') & (comp_annual['fyear'] >= 1988)) |
                             ((comp_annual['incorp'] == 'NV') & (comp_annual['fyear'] >= 1991)) |
                             ((comp_annual['incorp'] == 'NJ') & (comp_annual['fyear'] >= 1986)) |
                             ((comp_annual['incorp'] == 'NY') & (comp_annual['fyear'] >= 1985)) |
                             ((comp_annual['incorp'] == 'OK') & (comp_annual['fyear'] >= 1991)) |
                             ((comp_annual['incorp'] == 'OH') & (comp_annual['fyear'] >= 1990)) |
                             ((comp_annual['incorp'] == 'PA') & (comp_annual['fyear'] >= 1989)) |
                             ((comp_annual['incorp'] == 'RI') & (comp_annual['fyear'] >= 1990)) |
                             ((comp_annual['incorp'] == 'SC') & (comp_annual['fyear'] >= 1988)) |
                             ((comp_annual['incorp'] == 'SD') & (comp_annual['fyear'] >= 1990)) |
                             ((comp_annual['incorp'] == 'TN') & (comp_annual['fyear'] >= 1988)) |
                             ((comp_annual['incorp'] == 'VA') & (comp_annual['fyear'] >= 1988)) |
                             ((comp_annual['incorp'] == 'WA') & (comp_annual['fyear'] >= 1987)) |
                             ((comp_annual['incorp'] == 'WI') & (comp_annual['fyear'] >= 1987)) |
                             ((comp_annual['incorp'] == 'WY') & (comp_annual['fyear'] >= 1989)), 1, 0)

# Create log cash column for column 5 of Table 2. Based on what I was able to find in the G&M paper, there was no
# mention on how zero cash was dealt with so as to ensure the log of cash is well-defined. Because of this, I
# follow the same approach as above for assets and sales by removing firms with negative cash and additionally
# firms with zero cash. This leaves us with 149,823 rows.
comp_annual = comp_annual[comp_annual['ch'] > 0]
comp_annual['log_ch'] = np.log(comp_annual['ch'])


# ---------------------------------------- Clean Compustat Quarterly Data ---------------------------------------- #
# From WRDS go to: CRSP -> CRSP/Compustat Merged -> Fundamentals Quarterly. The variables we select are shown
# in the dataframe comp_quarterly below. We use this data in order to calculate the standard deviation of the
# quarterly cash flow ratio.

# Download quarterly CRSP/Compustat Merged (CCM) in order to create the cash flow volatility dependent variable.
# The variables we request are: GVKEY, fyearq, oiadpq, atq, actq, cheq, dlcq, dpq, and lctq. Refer to page 23, Table
# A.2, from G&M to see how the quarterly ratio of cash flow to assets is calculated, which warrants the inclusion of
# many of the listed variables.
comp_quarterly = pd.read_stata('compustat_quarterly.dta')
comp_quarterly.columns = comp_quarterly.columns.str.lower()
comp_quarterly = comp_quarterly[['gvkey', 'datadate', 'fyearq', 'oiadpq', 'atq', 'actq', 'cheq', 'dlcq', 'dpq', 'lctq']]
comp_quarterly = comp_quarterly.sort_values(by=['gvkey', 'datadate'])

# Create accruals column in order to compute quarterly ratio of cash flow to assets
# Step (1): Calculate quarterly change in atq
comp_quarterly['actq_lag'] = comp_quarterly.groupby(['gvkey'])['actq'].shift()
comp_quarterly['actq_change'] = comp_quarterly['actq'] - comp_quarterly['actq_lag']
# Step (2): Calculate quarterly change in cheq
comp_quarterly['cheq_lag'] = comp_quarterly.groupby(['gvkey'])['cheq'].shift()
comp_quarterly['cheq_change'] = comp_quarterly['cheq'] - comp_quarterly['cheq_lag']
# Step (3): Calculate quarterly change in lctq
comp_quarterly['lctq_lag'] = comp_quarterly.groupby(['gvkey'])['lctq'].shift()
comp_quarterly['lctq_change'] = comp_quarterly['lctq'] - comp_quarterly['lctq_lag']
# Step (4): Calculate quarterly change in dlcq
comp_quarterly['dlcq_lag'] = comp_quarterly.groupby(['gvkey'])['dlcq'].shift()
comp_quarterly['dlcq_change'] = comp_quarterly['dlcq'] - comp_quarterly['dlcq_lag']
# Step (5): Calculate accruals
comp_quarterly['accruals'] = comp_quarterly['actq_change'] - comp_quarterly['cheq_change'] \
                             - comp_quarterly['lctq_change'] + comp_quarterly['dlcq_change'] - comp_quarterly['dpq']

# Only keep accruals column
comp_quarterly = comp_quarterly[['gvkey', 'datadate', 'fyearq', 'oiadpq', 'atq', 'accruals']]
comp_quarterly = comp_quarterly[comp_quarterly.accruals.notna()]

# Create column for quarterly ratio of cash flow to assets. As done in G&M, we winsorize at the 1% level. Since
# this is the only financial ratio we add to our dataframe, this is the only winsorizing we perform.
comp_quarterly['atq_lag'] = comp_quarterly.groupby(['gvkey'])['atq'].shift()
comp_quarterly['cash_flow_ratio'] = (comp_quarterly['oiadpq'] - comp_quarterly['accruals']) / comp_quarterly['atq_lag']
comp_quarterly = comp_quarterly[comp_quarterly.cash_flow_ratio.notna()]
comp_quarterly['cash_flow_ratio'] = winsorize(comp_quarterly['cash_flow_ratio'], limits=(0.01, 0.01))

# Create column for cash flow volatility
comp_quart_annual_cf_vols = comp_quarterly.groupby(['gvkey', 'fyearq']).cash_flow_ratio.std().reset_index()
comp_quart_annual_cf_vols.rename(columns={'fyearq': 'fyear', 'cash_flow_ratio': 'cash_flow_vol'}, inplace=True)
comp_quart_annual_cf_vols = comp_quart_annual_cf_vols[comp_quart_annual_cf_vols.cash_flow_vol.notna()]

# After loading the below two additional dataframes, we will come back to this dataframe in order to merge it
# with the annual Compustat dataframe, comp_annual, created above.


# ------------------------------------------ Clean CRSP Daily Data ------------------------------------------ #
# From WRDS go to: CRSP -> Stock/Security Files -> Daily Stock File. Obtain RET and PERMCO only.

# Load in data
crsp_daily = pd.read_stata('crsp_daily.dta')

# Create fyear column which will allow us to merge with compustat. Need to convert calendar year to fiscal year
crsp_daily['month'] = pd.DatetimeIndex(crsp_daily['date']).month
crsp_daily['year'] = pd.DatetimeIndex(crsp_daily['date']).year
crsp_daily['fyear'] = np.where(crsp_daily['month'].isin([1, 2, 3, 4, 5, 6]), crsp_daily['year'] - 1, crsp_daily['year'])

# Keep only those columns we will need later on
crsp_daily = crsp_daily[['date', 'PERMCO', 'RET', 'fyear']]

# Get annual stock volatility.
# Step (1): Get sum of squared returns for each PERMCO for each fyear
crsp_daily['squared_RET'] = crsp_daily['RET'] ** 2
crsp_annual_stock_vol = crsp_daily.groupby(['PERMCO', 'fyear']).squared_RET.sum().reset_index()
crsp_annual_stock_vol.rename(columns={'squared_RET': 'sum_squared_RET'}, inplace=True)
# Step (2): Get number of trading days for each PERMCO for each fyear
crsp_annual_num_trading_days = crsp_daily.groupby(['PERMCO', 'fyear']).squared_RET.count().reset_index()
crsp_annual_num_trading_days.rename(columns={'squared_RET': 'trading_days'}, inplace=True)
# Step (3): Calculate stock volatility column as described in G&M; that is, calculate the square root of the
#           sum of squared daily returns and multiply this value by 252 divided by the number of days the
#           stock traded.
crsp_annual = pd.merge(crsp_annual_stock_vol, crsp_annual_num_trading_days, how='left', left_on=['PERMCO', 'fyear'],
                       right_on=['PERMCO', 'fyear'])
# Originally multiplied by 252 / num_trading_days outside of the square root ... the paper says raw sum, i.e. inside
# the square root!!
crsp_annual['stock_vol'] = (252 / crsp_annual['trading_days']) * np.sqrt(crsp_annual['sum_squared_RET'])

crsp_annual = crsp_annual[['PERMCO', 'fyear', 'stock_vol']]
print(crsp_annual.head(25))


# -------------------------------------------- CCM Annual Data -------------------------------------------- #
# From WRDS go to: CRSP -> CRSP/Compustat Merged -> Fundamentals Annual. The variables we select are shown
# in the dataframe ccm_annual below.

# Finally, we download annual CCM data in order to obtain a link between GVKEY and PERMCO. From this, we will be
# able to add a GVKEY column to the crsp_annual dataframe constructed above. After doing so, we will be able to
# merge our CRSP data with our annual compustat dataframe comp_annual.
#
# The only variables we require are GVKEY, LPERMCO, and FYEAR.

ccm_annual = pd.read_stata('ccm_annual.dta')[['GVKEY', 'LPERMCO', 'fyear']]
ccm_annual.columns = ccm_annual.columns.str.lower()


# ----------------------------------------------- Merge Data ---------------------------------------------- #

# Merge ccm_annual with compustat annual. This will allow us to merge our CRSP data since we now have a link
# between gvkey and permco for each year.
comp_annual_merge_w_ccm_annual = pd.merge(comp_annual, ccm_annual, how='left', left_on=['gvkey', 'fyear'],
                                          right_on=['gvkey', 'fyear'])


# -------------- Dataset for Column 4 Replication -------------- #

# Merge the above constructed (line 204) dataframe with our compustat quarterly dataframe. We will use this
# dataframe in order to replicate Column 4 from G&M.
cash_flow_vol_df = pd.merge(comp_annual_merge_w_ccm_annual, comp_quart_annual_cf_vols, how='left',
                            left_on=['gvkey', 'fyear'], right_on=['gvkey', 'fyear'])
cash_flow_vol_df = cash_flow_vol_df[cash_flow_vol_df.cash_flow_vol.notna()]


# ------------ Dataset for Columns 1 & 3 Replication ------------ #

# Merge the comp_annual_merge_w_ccm_annual dataframe with our crsp_annual dataframe. We will use this dataframe
# in order to replicate Columns 1 and 3 from G&M.
stock_and_op_asset_vol_df = pd.merge(comp_annual_merge_w_ccm_annual, crsp_annual, how='left', left_on=['lpermco', 'fyear'],
                                     right_on=['PERMCO', 'fyear'])
stock_and_op_asset_vol_df = stock_and_op_asset_vol_df[stock_and_op_asset_vol_df.stock_vol.notna()]

# Construct operating asset volatility column as explained on the bottom of page 22, in table A.2, from G&M. That is,
# the factor we multiply by stock volatility, E/(V-C), is given by multiplying csho and prcc_f and then dividing by
# lt plus the product of csho and prcc_f minus ch.
stock_and_op_asset_vol_df['E_by_V_minus_C'] = (stock_and_op_asset_vol_df['csho']*stock_and_op_asset_vol_df['prcc_f']) / \
                                              (stock_and_op_asset_vol_df['lt'] + (stock_and_op_asset_vol_df['csho'] *
                                                                                  stock_and_op_asset_vol_df['prcc_f']) -
                                               stock_and_op_asset_vol_df['ch'])
stock_and_op_asset_vol_df['op_asset_vol'] = stock_and_op_asset_vol_df['stock_vol'] * \
                                            stock_and_op_asset_vol_df['E_by_V_minus_C']
print(stock_and_op_asset_vol_df.shape)
stock_and_op_asset_vol_df = stock_and_op_asset_vol_df[stock_and_op_asset_vol_df.op_asset_vol.notna()]
print(stock_and_op_asset_vol_df.shape)

# -------------- Dataset for Column 5 Replication -------------- #

# We copy the comp_annual dataframe to a new dataframe whose name corresponds to the dependent variable log(cash) as
# this is the dataframe we will use to replicate Column 5 from G&M.
log_cash_df = comp_annual.copy()


# ---------------------------------- Replication of Columns 1, 3, 4, and 5 ---------------------------------- #

# ------------- Column 1 replication ------------- #
column1 = Regpyhdfe(df=stock_and_op_asset_vol_df, target='stock_vol',
                    predictors=['BC'], absorb_ids=['gvkeyn', 'styear', 'sicyear'], cluster_ids=['incorpn'])
results_column1 = column1.fit()
print(results_column1.summary2())

# ------------- Column 3 replication ------------- #
column3 = Regpyhdfe(df=stock_and_op_asset_vol_df, target='op_asset_vol',
                    predictors=['BC'], absorb_ids=['gvkeyn', 'styear', 'sicyear'], cluster_ids=['incorpn'])
results_column3 = column3.fit()
print(results_column3.summary2())

# ------------- Column 4 replication ------------- #
column4 = Regpyhdfe(df=cash_flow_vol_df, target='cash_flow_vol',
                    predictors=['BC'], absorb_ids=['gvkeyn', 'styear', 'sicyear'], cluster_ids=['incorpn'])
results_column4 = column4.fit()
print(results_column4.summary2())

# ------------- Column 5 replication ------------- #
column5 = Regpyhdfe(df=log_cash_df, target='log_ch',
                    predictors=['BC'], absorb_ids=['gvkeyn', 'styear', 'sicyear'], cluster_ids=['incorpn'])
results_column5 = column5.fit()
print(results_column5.summary2())



