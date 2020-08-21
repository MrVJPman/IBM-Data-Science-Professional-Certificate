import numpy as np  
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt

survey_dataframe = pd.read_csv("Topic_Survey_Assignment.csv", index_col = 0)
survey_dataframe.sort_values(['Very interested'], ascending=False, axis=0, inplace=True)
survey_dataframe = round(survey_dataframe/2233, 2)  
survey_dataframe.plot(kind='bar', 
                      figsize=(20, 8),
                      fontsize = 14, 
                      width = 0.8,
                      color = ['#5cb85c', '#5bc0de', '#d9534f'])

index = 0
for column in survey_dataframe.transpose():     
    for percentage in survey_dataframe.transpose()[column]:
        plt.annotate(str(percentage), xy=(-0.35 + index, percentage))
        index = index + 0.25
    index = index + 0.25
    
    
plt.title("Percentage of Respondent's interest in Data Science Areas", fontdict = {'fontsize' : 16}) # add title to the plot
#plt.show()

#print(survey_dataframe)



#==================================================
import folium
san_francisco_crime_dataframe = pd.read_csv("Police_Department_Incidents_-_Previous_Year__2016_.csv")

neighbourhood_crime_count_dataframe = san_francisco_crime_dataframe["PdDistrict"].value_counts().to_frame()
neighbourhood_crime_count_dataframe.reset_index(inplace = True) 
neighbourhood_crime_count_dataframe.rename(columns={'PdDistrict': 'Count', "index":"Neighborhood"}, inplace=True)

san_francisco_geo = 'san-francisco.geojson'


world_map = folium.Map(location=[37.77, -122.42], zoom_start=12)
world_map.choropleth(
    geo_data=san_francisco_geo,
    data=neighbourhood_crime_count_dataframe,
    columns=['Neighborhood', 'Count'],
    key_on="feature.properties.DISTRICT",
    fill_color='YlOrRd', 
    fill_opacity=0.7, 
    line_opacity=0.2,
    legend_name='Crime Rate in San Franciso'
)

world_map.save("world_map.html")


#import webbrowser
#webbrowser.open_tab('world_map.html')