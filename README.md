## Predict non-medical calls to the San Francisco Fire Department

This project generates a model to forecast the next day number of non-medical calls to the San Francisco Fire Department, allowing the Department to get ready with the necessary resources and personnel to take care of the calls.

The calls could be related to fire, rescues, biological-hazards, explotions, industrial accidents, etc. 

Information comes from different sources:
- __Calls to SFFD:__ More than 500k records with details about every call since 2003 ([DataSF](https://data.sfgov.org/Public-Safety/Fire-Incidents/wr8u-xric) : Downloaded in July 28th 2019).
- __Weather conditions:__ [NOAA](https://www.ncdc.noaa.gov/cdo-web/)
    - Daily information for precipitation, minimum and maximum temperature from SF downtown station
    - Daily information for: average wind and gusts (2mins and 5 secs) from SF airport
 
Data from the calls to the SFFD was aggregated by day and merged with weather data. Creation of the time series model involved __feature engineering__ like:
- __RainxGusts =__ (precipitation x gust at 5 seconds)<sup>2</sup>
- __Moving holidays__ such as thanksgiving

The project was developped in Python 3 using libraries such as: Pandas, Sklearn and ipywidgets

Download the Jupyter notebook to play with this interactive tool that forecasts the calls to SFFD distpatch and enjoy!
