import matplotlib.pyplot as plt
import numpy as np

weather_filename = 'fort_lauderdale.csv'
weather_file = open(weather_filename)
weather_data = weather_file.read()
weather_file.close()

#print(len(weather_data))
#print(weather_data[:200])

# Break the weather records into lines
lines = weather_data.split('\n')

#print(len(lines))
#for i in range(5):
#    print(lines[i])

labels = lines[0]
values = lines[1:]
n_values = len(values)

#print(labels)
#for i in range(10):
#    print(values[i])

# Break the list of comma-separated values strings into list of values
year = []
month = []
day = []
max_temp = []
j_year = 1
j_month = 2
j_day = 3
j_max_temp = 5

for i_row in range(n_values):
    split_values = values[i_row].split(',')
    #print(split_values)
    if len(split_values) >= j_max_temp:
        year.append(int(split_values[j_year]))
        month.append(int(split_values[j_month]))
        day.append(int(split_values[j_day]))
        max_temp.append(float(split_values[j_max_temp]))


#plt.plot(max_temp)
#plt.show()

# Isolate the recent data.
#i_mid = len(max_temp) // 2
#temps = np.array(max_temp[i_mid:])
#temps[np.where(temps == -99.9)] = np.nan

#plt.plot(temps, color='black', marker='.', linestyle='none')
#plt.show()

# Remove all the nan's
# Trim both ends and fill nans on the middle
#print(np.where(np.isnan(temps))[0])
#print(np.where(np.logical_not(np.isnan(temps)))[0][0])

#i_start = np.where(np.logical_not(np.isnan(temps)))[0][0]
#temps = temps[i_start:]
#print(np.where(np.isnan(temps))[0])
#i_nans = np.where(np.isnan(temps))[0]
#print(np.diff(i_nans))

# Replace all nans with the most recent non-nan
#for i in range(temps.size):
#    if np.isnan(temps[i]):
#        temps[i] = temps[i - 1]


# Isolate the recent data.
i_mid = len(max_temp) // 2
temps = np.array(max_temp[i_mid:])
year = year[i_mid:]
month = month[i_mid:]
day = day[i_mid:]
temps[np.where(temps == -99.9)] = np.nan

# Remove all the nans.
# Trim both ends and fill nans in the middle.
# Find the first non-nan.
i_start = np.where(np.logical_not(np.isnan(temps)))[0][0]
temps = temps[i_start:]
year = year[i_start:]
month = month[i_start:]
day = day[i_start:]
i_nans = np.where(np.isnan(temps))[0]

# Replace all nans with the most recent non-nan.
for i in range(temps.size):
    if np.isnan(temps[i]):
        temps[i] = temps[i - 1]

#plt.plot(temps)
#plt.show()
# data set cleaned!

# Determine whether the previous day's temperature is related to the following day.

#plt.plot(temps[:-1], temps[1:], color='black', marker='.', linestyle='none')
#plt.show()


def scatter(x, y):
    """
    Make a scatter plot with jitter
    :param x: array containing numbers
    :param y: array containing numbers
    :return: plots a scatter plot with jitter
    """
    x_jitter = x + np.random.normal(size=x.size, scale=.5)
    y_jitter = y + np.random.normal(size=y.size, scale=.5)
    plt.plot(
        x_jitter, y_jitter,
        color='black',
        marker='.',
        linestyle='none',
        alpha=.1
    )
    plt.show()


shift = 1
#scatter(temps[:-shift], temps[shift:])

#print(np.corrcoef(temps[:-shift], temps[shift:]))

autocorr = []
for shift in range(1, 1000):
    correlation = np.corrcoef(temps[:-shift], temps[shift:])[1, 0]
    autocorr.append(correlation)

#plt.plot(autocorr)
#plt.show()

## Create 10-day medians for each of the year
## We only have 19 years of data using a 10 day range gives us a 190 day
day_of_year = np.zeros(temps.size)


def find_day_of_year(year, month, day):
    """
    Convert year, month, date to day of the year
    :param year: int
    :param month: int
    :param day: int
    :return: day_of_year: int
    """
    days_per_month = np.array([
        31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
    ])
    if year % 4 == 0:
        days_per_month[1] += 1

    day_of_year = np.sum(np.array(
        days_per_month[:month - 1]
    )) + day - 1
    return day_of_year


for i_row in range(temps.size):
    day_of_year[i_row] = find_day_of_year(
        year[i_row], month[i_row], day[i_row]
    )

#scatter(day_of_year, temps)

median_temp_calender = np.zeros(366)
ten_day_medians = np.zeros(temps.size)
for i_day in range(0, 365):
    low_day = i_day - 5
    high_day = i_day + 4
    if low_day < 0:
        low_day += 365
    if high_day > 365:
        high_day += -365
    if low_day < high_day:
        i_window_days = np.where(
            np.logical_and(day_of_year >= low_day,
                           day_of_year <= high_day)
            )
    else:
        i_window_days = np.where(
            np.logical_or(day_of_year >= low_day,
                          day_of_year <= high_day))
    ten_day_median = np.median(temps[i_window_days])
    median_temp_calender[i_day] = ten_day_median
    ten_day_medians[np.where(day_of_year == i_day)] = ten_day_median
    if i_day == 364:
        ten_day_medians[np.where(day_of_year == 365)] = ten_day_median
        median_temp_calender[365] = ten_day_median

#print(ten_day_medians.size, np.unique(ten_day_medians), ten_day_medians)

#scatter(ten_day_medians, temps)
#plt.plot(temps)
#plt.plot(ten_day_medians)
#plt.show()


def predict(day, month, year, temperature_calender):
    """
    For a given day, month, and year, predict the high temperature for the
    Fort Lauderdale beach.
    :param day: int
    :param month: int
    :param year: int
    :param temperature_calender: array of floats
        The typical temperature for each day of the year.
    :return:
    prediction: float
    """
    doy = find_day_of_year(year, month, day)
    prediction = temperature_calender[doy]
    return prediction


test_day = 1
test_year = 2015
test_month = 6
prediction = predict(test_year, test_month, test_day, median_temp_calender)
print(test_year, test_month, test_day, prediction)
