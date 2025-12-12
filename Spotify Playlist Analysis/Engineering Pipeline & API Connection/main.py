#first install packages- pip3 install python-dotenv AND pip3 install requests

from dotenv import load_dotenv
import os 
import base64
import requests
import json
from pprint import pprint
from typing import TypedDict
import math


load_dotenv()

client_id = os.getenv("client_id", "")
client_secret = os.getenv("client_secret", "")

def get_token():
    auth_string = client_id + ":" + client_secret
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes),"utf-8") 

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": "Basic " + auth_base64,
        "Content-Type": "application/x-www-form-urlencoded"
    } 
    data = {"grant_type": "client_credentials"} 
    result = requests.post(url,headers = headers, data = data)
    json_result = json.loads(result.content)
    token = json_result["access_token"] 
    return token

def get_auth_header(token):
    return {"Authorization": "Bearer " + token}

def get_playlist_items(token, playlist_id):
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    headers = get_auth_header(token)
    result = requests.get(url,headers=headers)
    json_result = json.loads(result.content)["items"]

    # return json_result


class PlaylistTracksReponse(TypedDict):
    href: str
    limit: int
    next: str
    offset: int
    previous: str
    total: int
    items: list[dict]

#https://api.spotify.com/v1/playlists/3cEYpjA9oz9GiPac4AsH4n/tracks?offset=5&limit=100&locale=en-US,en;q=0.9

def get_playlist_tracks(token, playlist_id: str, offset: int = 0, limit: int = 100) -> PlaylistTracksReponse:
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks?offset={offset}&limit={limit}"
    headers = get_auth_header(token)
    result = requests.get(url,headers=headers)
    json_result = json.loads(result.content)

    return json_result

token = get_token()

def get_all_tracks(token, playlist_id: str, offset: int = 0, limit: int = 100):
    initial_tracks = get_playlist_tracks(token, playlist_id, offset, limit)
    total = initial_tracks['total']
    
    iterations = math.ceil(total/limit)
    
    tracks_list = []
    for iteration in range(iterations):
        factor = iteration * limit + 1
        tracks = get_playlist_tracks(token, "5FLAvRkSh7iqGqaaazMPuK", factor)
        for item in tracks['items']:
            tracks_list.append(item['track'])
            #print(item['track']['name'])
    quit()
    return tracks_list

#y = get_all_tracks(token,"5FLAvRkSh7iqGqaaazMPuK", 0, 100)

def get_spotify_id(song_name, token):
    encoded_song_name = song_name.replace(" ", "%20")
    response = requests.get(
        f"https://api.spotify.com/v1/search?q={encoded_song_name}&type=track",
        headers={"Authorization": f"Bearer {token}"})
    spotify_id = response.json()["tracks"]["items"][0]["id"]
    return spotify_id


def get_track_ids(token,tracks_list):
    tracks_id_list = []
    for song in tracks_list:
        song_id = get_spotify_id(song,token)
        tracks_id_list.append(song_id)
        print(song_id)
    quit()
    return tracks_id_list

tracks_list = get_all_tracks(token,"5FLAvRkSh7iqGqaaazMPuK", 0, 100)
print(tracks_list)
get_track_ids(token,tracks_list)

# for item in y:
#     print(item['name'])
# pprint(y)

# print(my_playlist_items[3].keys())
# tracks = get_playlist_tracks(token, "5FLAvRkSh7iqGqaaazMPuK")
# tracks = get_playlist_tracks(token, "3ZFmXY7eckrAiyISdCsbUW", 0, 50)

# print(tracks["items"][1]['track']['name'])
# count = 0

# print(len(tracks['items']))
# for item in tracks['items']:
#     count += 1
#     print(item['track']['name'])
#     quit()

# print(count)
# duration = 0
# for item in tracks['items']:
#     duration += item['track']['duration_ms']

# print(duration/60000)




## create TypedDict annotation for this sample spotify API response
