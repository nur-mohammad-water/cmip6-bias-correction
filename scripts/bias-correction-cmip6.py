import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt

# ---------------- SETUP ----------------
fold_dir = 'C:/Users/Thotapit/OneDrive - Hereon/Documents/Moratuwa MSc/02. Data/'
yy = 6.68
xx = 80.4

# ---------------- OBSERVED DATA ----------------
rf_obs = pd.read_excel(fold_dir + 'Rathnapura_Rf.xlsx')
rf_obs.set_index('Date', inplace=True)
rf_obs = rf_obs[(rf_obs.index >= '1986-01-01') & (rf_obs.index <= '2014-12-31')]

rf_obs_month = rf_obs.resample('ME').sum()
rf_obs_month.plot()

rf_obs_meanmon = rf_obs_month.groupby(rf_obs_month.index.month).mean()
rf_obs_meanmon.plot()

rf_obs_stdmon = rf_obs_month.groupby(rf_obs_month.index.month).std()

# ---------------- HISTORICAL GCM DATA ----------------
rf_hissim = xr.open_dataset(fold_dir + 'pr_day_CNRM-CM6-1-HR_historical_r1i1p1f2_gr_19500101-20141231.nc')

rf_hissim['pr'][0, :, :].plot()

rf_his_sim = rf_hissim['pr'].sel(lat=yy, lon=xx, method='nearest')
rf_his_sim = rf_his_sim.sel(time=slice("1986-01-01", "2014-12-31")) * 86400
rf_his_sim = rf_his_sim.to_dataframe()

rf_sim_mon = rf_his_sim['pr'].resample('ME').sum()
rf_sim_meanmon = rf_sim_mon.groupby(rf_sim_mon.index.month).mean()
rf_sim_stdmon = rf_sim_mon.groupby(rf_sim_mon.index.month).std()

plt.figure()
plt.plot(rf_sim_meanmon, label='Simulated')
plt.plot(rf_obs_meanmon, label='Observed')
plt.xlabel('Month')
plt.ylabel('Monthly Rainfall (mm)')
plt.legend()
plt.savefig('Monthly_Rainfall.jpg')

# ---------------- MEAN-BASED BIAS CORRECTION FACTORS ----------------
bias_fac = rf_obs_meanmon['Rathnapura'].to_numpy() / rf_sim_meanmon.to_numpy()

# --- Apply to historical data ---
rf_his_sim['month'] = rf_his_sim.index.month
rf_his_sim['BiasFac'] = rf_his_sim['month'].apply(lambda x: bias_fac[x - 1])
rf_his_sim['Correct_pr'] = rf_his_sim['pr'] * rf_his_sim['BiasFac']

plt.figure()
rf_his_sim['pr'].plot()
rf_his_sim['Correct_pr'].plot()

# --- Apply to future data ---
rf_fursim = xr.open_dataset(fold_dir + "pr_day_CNRM-CM6-1-HR_ssp585_r1i1p1f2_gr_20150101-21001231.nc")

rf_fur_sim = rf_fursim['pr'].sel(lat=yy, lon=xx, method='nearest') * 86400
rf_fur_sim = rf_fur_sim.to_dataframe()

rf_fur_sim['month'] = rf_fur_sim.index.month
rf_fur_sim['BiasFac'] = rf_fur_sim['month'].apply(lambda x: bias_fac[x - 1])
rf_fur_sim['Correct_pr'] = rf_fur_sim['pr'] * rf_fur_sim['BiasFac']

# ---------------- COMPARISON PLOT: OBSERVED vs RAW vs CORRECTED (monthly) ----------------
rf_hiscorr_mon = rf_his_sim['Correct_pr'].resample('ME').sum()
rf_hiscorr_meanmon = rf_hiscorr_mon.groupby(rf_hiscorr_mon.index.month).mean()

plt.figure()
plt.plot(rf_obs_meanmon, label='Observed')
plt.plot(rf_sim_meanmon, label='Raw GCM')
plt.plot(rf_hiscorr_meanmon, label='Bias-Corrected GCM')
plt.xlabel('Month')
plt.ylabel('Monthly Rainfall (mm)')
plt.legend()
plt.savefig('Comparison_Monthly_Rainfall.jpg')

# ---------------- PERFORMANCE EVALUATION ----------------
def evaluate(obs, sim):
    obs = np.array(obs).flatten()
    sim = np.array(sim).flatten()
    rmse = np.sqrt(np.mean((obs - sim) ** 2))
    bias = np.mean(sim - obs)
    corr = np.corrcoef(obs, sim)[0, 1]
    return {'RMSE': rmse, 'Bias': bias, 'Correlation': corr}

raw_perf = evaluate(rf_obs_meanmon['Rathnapura'].to_numpy(), rf_sim_meanmon.to_numpy())
corrected_perf = evaluate(rf_obs_meanmon['Rathnapura'].to_numpy(), rf_hiscorr_meanmon.to_numpy())

print("Raw GCM performance:", raw_perf)
print("Bias-corrected GCM performance:", corrected_perf)

# ---------------- VARIANCE-BASED BIAS CORRECTION ----------------
# X_O' = ((X_M' - mu_M) / sigma_M) * sigma_O + mu_O

def variance_bias_correct(df, value_col, mu_obs, sigma_obs, mu_sim, sigma_sim):
    df = df.copy()
    df['month'] = df.index.month
    df['mu_M'] = df['month'].apply(lambda x: mu_sim[x - 1])
    df['sigma_M'] = df['month'].apply(lambda x: sigma_sim[x - 1])
    df['mu_O'] = df['month'].apply(lambda x: mu_obs[x - 1])
    df['sigma_O'] = df['month'].apply(lambda x: sigma_obs[x - 1])
    df['Corrected_var'] = ((df[value_col] - df['mu_M']) / df['sigma_M']) * df['sigma_O'] + df['mu_O']
    df['Corrected_var'] = df['Corrected_var'].clip(lower=0)
    return df

mu_obs_arr = rf_obs_meanmon['Rathnapura'].to_numpy()
sigma_obs_arr = rf_obs_stdmon['Rathnapura'].to_numpy()
mu_sim_arr = rf_sim_meanmon.to_numpy()
sigma_sim_arr = rf_sim_stdmon.to_numpy()

rf_his_sim = variance_bias_correct(rf_his_sim, 'pr', mu_obs_arr, sigma_obs_arr, mu_sim_arr, sigma_sim_arr)
rf_fur_sim = variance_bias_correct(rf_fur_sim, 'pr', mu_obs_arr, sigma_obs_arr, mu_sim_arr, sigma_sim_arr)

plt.figure()
rf_his_sim['pr'].plot(label='Raw')
rf_his_sim['Correct_pr'].plot(label='Mean-Corrected')
rf_his_sim['Corrected_var'].plot(label='Variance-Corrected')
plt.legend()
plt.savefig('Variance_Bias_Correction.jpg')

plt.show()
