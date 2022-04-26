import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def serpent_chart(data, season="year", direction="outwards"):

    data = data.reset_index(drop=True)

    # Translate data
    data["ts"] = pd.to_datetime(data["ts"])
    r = pd.Series(data.index)
    delta = data["y"]
    delta = delta/(2*delta.abs().max())

    # Check datapoints per year
    data["year"] = pd.DatetimeIndex(data["ts"]).year
    avg_data_year = data[["year", "ts"]].groupby(["year"]).agg(["count"]).reset_index(drop=True).agg("mean").iloc[0]
    data = data.drop(["year"], axis=1)
    if(avg_data_year <= 12):
        data_points = 12
        data_unit = np.timedelta64(1,'M')
    else:
        data_points = 365 # FIX - Account for leap year
        data_unit = np.timedelta64(1,'D')

    # Translate seasonality and add offset
    if(season=="year"):
        season_ratio = 1/data_points

        min_date = data["ts"].min()
        offset = int((min_date - pd.to_datetime("01-01-"+str(min_date.year)))/data_unit)

        rlabels = np.sort(np.unique(pd.DatetimeIndex(data["ts"]).year.to_numpy()))

    r = r+offset
    r = r*season_ratio

    # Calcualte theta
    theta = (-2 * np.pi * r) + (np.pi/2)

    # Invert direction
    if(direction=="inwards"):
        revert_r = r.copy()

        for i in range(len(revert_r)):
            revert_r[i] = revert_r[i] - r[i] + r[len(r)- 1 -i]

        r = revert_r.copy()

        rlabels = rlabels[::-1]

    # Calculate Boundaries
    delta_max = r + delta
    delta_min = r - delta

    # Create Thetagrids
    if(season=="year"):
        thetagrids_pos = np.arange(0, 2*np.pi, np.pi/6) * (180/np.pi)
        thetagrids_labels = ["April", "March", "February", "January", "December", "November", "October", "September", "August", "July", "June", "May"]

    # Create plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta, r)
    ax.fill_between(theta ,delta_max, delta_min, alpha=0.5)

    # Set thetagrids
    ax.set_thetagrids(thetagrids_pos, thetagrids_labels)

    # Set season labels
    ax.set_rgrids(range(len(rlabels)), labels=rlabels, angle=90)
    ax.grid(True)

    return fig, ax

# Gaussian noise
r = pd.date_range("2018-06-01", "2022-01-09")
noise = np.random.normal(0.1, 1, len(r))

data = pd.DataFrame(data={"ts":r, "y": noise})

fig, ax = serpent_chart(data)

fig.suptitle("Gaussian Noise (Serpent Chart)")
fig.savefig("plots/noise.jpg")
plt.close()

plt.figure()
data_series = data["y"]
data_series.index = pd.DatetimeIndex(data["ts"]).to_pydatetime()
data_series = pd.to_numeric(data_series, errors="coerce")
data_series.sort_values()
data_series.plot()
plt.title("Gaussian Noise (Linear Plot)")
plt.savefig("linear_plots/noise.jpg")

# Australia low temperature
data = pd.read_csv("datasets/australia_temp.csv")
data = data.tail(1500)

fig, ax = serpent_chart(data)

fig.suptitle("Australia Lowest Temperature (Serpent Chart)")
fig.savefig("plots/australia_temp.jpg")
plt.close()

plt.figure()
data_series = data["y"]
data_series.index = pd.DatetimeIndex(data["ts"]).to_pydatetime()
data_series = pd.to_numeric(data_series, errors="coerce")
data_series.sort_values()
data_series.plot()
plt.title("Australia Lowest Temperature (Linear Plot)")
plt.savefig("linear_plots/australia_temp.jpg")

# Female births
data = pd.read_csv("datasets/female_births.csv")

fig, ax = serpent_chart(data)

fig.suptitle("Female Births (Serpent Chart)")
fig.savefig("plots/fem_births.jpg")
plt.close()

plt.figure()
data_series = data["y"]
data_series.index = pd.DatetimeIndex(data["ts"]).to_pydatetime()
data_series = pd.to_numeric(data_series, errors="coerce")
data_series.sort_values()
data_series.plot()
plt.title("Female Births (Linear Plot)")
plt.savefig("linear_plots/fem_births.jpg")

# Monthly Sunspots
data = pd.read_csv("datasets/monthly-sunspots.csv")
data["ts"] = pd.to_datetime(data["ts"], format="%Y-%m")
data = data.tail(60)

fig, ax = serpent_chart(data)

fig.suptitle("Sunspots (Serpent Chart)")
fig.savefig("plots/monthly_sunspots.jpg")
plt.close()

plt.figure()
data_series = data["y"]
data_series.index = pd.DatetimeIndex(data["ts"]).to_pydatetime()
data_series = pd.to_numeric(data_series, errors="coerce")
data_series.sort_values()
data_series.plot()
plt.title("Sunspots (Linear Plot)")
plt.savefig("linear_plots/monthly_sunspots.jpg")

#Shampoo Sales
data = pd.read_csv("datasets/shampoo.csv")
data["ts"] = pd.to_datetime(data["ts"], format="%Y-%m")
data = data.tail(60)

fig, ax = serpent_chart(data)

fig.suptitle("Shampoo Sales (Serpent Chart)")
fig.savefig("plots/shampoo.jpg")
plt.close()

plt.figure()
data_series = data["y"]
data_series.index = pd.DatetimeIndex(data["ts"]).to_pydatetime()
data_series = pd.to_numeric(data_series, errors="coerce")
data_series.sort_values()
data_series.plot()
plt.title("Shampoo Sales (Linear Plot)")
plt.savefig("linear_plots/shampoo.jpg")
