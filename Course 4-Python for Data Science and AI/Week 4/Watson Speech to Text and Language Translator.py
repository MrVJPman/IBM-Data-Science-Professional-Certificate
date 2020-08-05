#you will need the following library 
#Code from Juptyer Notebook
#!pip install ibm_watson wget

#First we import SpeechToTextV1 from ibm_watson.
#https://cloud.ibm.com/apidocs/speech-to-text?code=python
from ibm_watson import SpeechToTextV1 
import json
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

#The service endpoint is based on the location of the service instance, we store the information in the variable URL. 
#To find out which URL to use, view the service credentials.
url_s2t = "https://api.us-south.speech-to-text.watson.cloud.ibm.com/instances/2344fe0f-e3d2-49ab-862b-57ec806a39c3"
#retrieved from IBM Watson Service Page

#You require an API key, and you can obtain the key on the Dashboard .
iam_apikey_s2t = "JSwdI01S9dV9--To9WPI0-_m18fmg4vZ4pNgz-1X4Unk"
#retrieved from IBM Watson Service Page


#You create a Speech To Text Adapter object the parameters are the endpoint and API key.
authenticator = IAMAuthenticator(iam_apikey_s2t)
s2t = SpeechToTextV1(authenticator=authenticator)
s2t.set_service_url(url_s2t)

#Code from Juptyer Notebook
#!wget -O PolynomialRegressionandPipelines.mp3  https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/labs/PolynomialRegressionandPipelines.mp3

#We have the path of the wav file we would like to convert to text
filename='PolynomialRegressionandPipelines.mp3'

#We create the file object wav with the wav file using open ; we set the mode to "rb" , 
#this is similar to read mode, but it ensures the file is in binary mode.
#We use the method recognize to return the recognized text. 
#The parameter audio is the file object wav, the parameter content_type is the format of the audio file.
with open(filename, mode="rb")  as wav:
    response = s2t.recognize(audio=wav, content_type='audio/mp3')

#The attribute result contains a dictionary that includes the translation:
from pandas.io.json import json_normalize

json_normalize(response.result['results'],"alternatives")

print(response)

#We can obtain the recognized text and assign it to the variable recognized_text:
recognized_text=response.result['results'][0]["alternatives"][0]["transcript"]
type(recognized_text)

#====================================================================
#Language Translator

#First we import LanguageTranslatorV3 from ibm_watson.
#https://cloud.ibm.com/apidocs/speech-to-text?code=python
from ibm_watson import LanguageTranslatorV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

#The service endpoint is based on the location of the service instance, we store the information in the variable URL. 
#To find out which URL to use, view the service credentials.
url_lt='https://api.us-south.language-translator.watson.cloud.ibm.com/instances/4a781a58-6ff0-4301-a285-1d9a266e46c2'

#You require an API key, and you can obtain the key on the Dashboard.
apikey_lt='xlfzZzEEoeTBoIMUJZTqzTMIY6tGpaLfwZjY4QweX-bh'

#API requests require a version parameter that takes a date in the format version=YYYY-MM-DD.
#This lab describes the current version of Language Translator, 2018-05-01
version_lt='2018-05-01'

#we create a Language Translator object language_translator:
authenticator = IAMAuthenticator(apikey_lt)
language_translator = LanguageTranslatorV3(version=version_lt,authenticator=authenticator)
language_translator.set_service_url(url_lt)

#We can get a Lists the languages that the service can identify. 
#The method Returns the language code. For example English (en) to Spanis (es) and name of each language.
from pandas.io.json import json_normalize

json_normalize(language_translator.list_identifiable_languages().get_result(), "languages")

#https://labs.cognitiveclass.ai/tools/jupyterlab/lab/tree/labs/PY0101EN/PY0101EN-5.2_API_2.ipynb
#This code end here as there were some issues with the instructions