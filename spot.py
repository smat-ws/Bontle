import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

username = os.getenv('SPOTIFY_USERNAME', '')
clientID = os.getenv('SPOTIPY_CLIENT_ID', '')
clientSecret = os.getenv('SPOTIPY_CLIENT_SECRET', '')
redirect_uri = os.getenv('SPOTIPY_REDIRECT_URI', 'http://localhost:8888/callback')

def spotify_authenicate(client_id, client_secret, redirect_uri, username):
    if not client_id or not client_secret:
        print("Warning: Spotify credentials not configured. Spotify features will be disabled.")
        return None
    scope = "user-read-currently-playing user-modify-playback-state"
    auth_manager = SpotifyOAuth(client_id, client_secret, redirect_uri, scope=scope, username=username)
    return spotipy.Spotify(auth_manager = auth_manager)

# Only initialize Spotify if credentials are available
spotify = None
if clientID and clientSecret:
    try:
        spotify = spotify_authenicate(clientID, clientSecret, redirect_uri, username)
    except Exception as e:
        print(f"Failed to initialize Spotify: {e}")
        spotify = None

def get_current_playing_info():
    global spotify
    if spotify is None:
        return "Spotify not configured"
    
    current_track = spotify.current_user_playing_track()
    if current_track is None:
        return None
    
    artist_name = current_track['item']['artists'][0]['name']
    album_name = current_track['item']['album']['name']
    track_title = current_track['item']['name']

    return {
        "artist": artist_name,
        "album": album_name,
        "title": track_title
    }


def start_music():
    global spotify
    if spotify is None:
        return "Spotify not configured"
    try:
        spotify.start_playback()
    except spotipy.SpotifyException as e:
        return f"Error in starting playback: {str(e)}"
    
def stop_music():
    global spotify
    if spotify is None:
        return "Spotify not configured"
    try:
        spotify.pause_playback()
    except spotipy.SpotifyException as e:
        return f"Error in starting playback: {str(e)}"
    
def skip_to_next():
    global spotify
    if spotify is None:
        return "Spotify not configured"
    try:
        spotify.next_track()
    except spotipy.SpotifyException as e:
        return f"Error in starting playback: {str(e)}"
    
def skip_to_previous():
    global spotify
    if spotify is None:
        return "Spotify not configured"
    try:
        spotify.previous_track()
    except spotipy.SpotifyException as e:
        return f"Error in starting playback: {str(e)}"
    
