library(ARPALData)
library(tidyverse)

# download stations registry
reg <- ARPALData::get_ARPA_Lombardia_AQ_registry()

pollutants <- c('NO2', 'NOx', 'PM10', 'PM2.5', 'Ammonia')

for (pollutant in pollutants) {

  # download hourly data, from 2016 to 2021 of every station
  data <- ARPALData::get_ARPA_Lombardia_AQ_data(ID_station = NULL,
                                                Frequency = "daily",
                                                Var_vec = pollutant,
                                                Fns_vec = c("mean"),
                                                Year = 2016:2020)

  # remove 'NameStation' column to lighten the data
  data <- data %>% select(-NameStation)

  # export pollutant data csv file
  write.csv(data, file = paste(pollutant, '.csv', sep=''))

}
