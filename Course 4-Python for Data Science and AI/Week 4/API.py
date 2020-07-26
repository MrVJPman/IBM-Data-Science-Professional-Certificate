def one_dict(list_dict):
    keys=list_dict[0].keys()
    out_dict={key:[] for key in keys}
    for dict_ in list_dict:
        for key, value in dict_.items():
            out_dict[key].append(value)
    return out_dict  

import pandas as pd
import matplotlib.pyplot as plt

dict_={'a':[11,21,31],'b':[12,22,32]}
df=pd.DataFrame(dict_)
type(df)
df.head()
df.mean()

from nba_api.stats.static import teams
import matplotlib.pyplot as plt

#The method get_teams() returns a list of dictionaries the dictionary key id has a unique identifier for each team as a value
nba_teams = teams.get_teams()

#To make things easier, we can convert the dictionary to a table. 
#First, we use the function one dict, to create a dictionary. 
#We use the common keys for each team as the keys, the value is a list; 
#each element of the list corresponds to the values for each team. 
#We then convert the dictionary to a dataframe, each row contains the information for a different team.
dict_nba_team=one_dict(nba_teams)
df_teams=pd.DataFrame(dict_nba_team)
df_teams.head()

#Will use the team's nickname to find the unique id, 
#we can see the row that contains the warriors by using the column nickname as follows:
df_warriors=df_teams[df_teams['nickname']=='Warriors']

#we can use the following line of code to access the first column of the dataframe:
id_warriors=df_warriors[['id']].values[0][0]
#we now have an integer that can be used   to request the Warriors information 

#The function "League Game Finder " will make an API call, its in the module stats.endpoints
from nba_api.stats.endpoints import leaguegamefinder

# Since https://stats.nba.com does lot allow api calls from Cloud IPs and Skills Network Labs uses a Cloud IP.
# The following code is comment out, you can run it on jupyter labs on your own computer.
# gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=id_warriors)

# Since https://stats.nba.com does lot allow api calls from Cloud IPs and Skills Network Labs uses a Cloud IP.
# The following code is comment out, you can run it on jupyter labs on your own computer.
# gamefinder.get_json()

# Since https://stats.nba.com does lot allow api calls from Cloud IPs and Skills Network Labs uses a Cloud IP.
# The following code is comment out, you can run it on jupyter labs on your own computer.
# games = gamefinder.get_data_frames()[0]
# games.head()

#Code for Juptyer Notebook
#! wget https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/Chapter%205/Labs/Golden_State.pkl

file_name = "Golden_State.pkl"
games = pd.read_pickle(file_name)
games.head()

#We can create two dataframes, 
#one for the games that the Warriors faced the raptors at home and 
#the second for away games.
games_home=games [games ['MATCHUP']=='GSW vs. TOR']
games_away=games [games ['MATCHUP']=='GSW @ TOR']

#We can calculate the mean for the column PLUS_MINUS for the dataframes games_home and  games_away:
games_home.mean()['PLUS_MINUS']
games_away.mean()['PLUS_MINUS']

#We can plot out the PLUS MINUS column for for the dataframes games_home and  games_away. 
#We see the warriors played better at home.
fig, ax = plt.subplots()
games_away.plot(x='GAME_DATE',y='PLUS_MINUS', ax=ax)
games_home.plot(x='GAME_DATE',y='PLUS_MINUS', ax=ax)
ax.legend(["away", "home"])
plt.show()