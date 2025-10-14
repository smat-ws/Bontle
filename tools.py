
import asyncio
import assist
# from icrawler.builtin import GoogleImageCrawler
import os
import spot

async def parse_command(command):

    if "play" in command:
        result = spot.start_music()
        if result and "not configured" in result:
            response = "Sorry, Spotify is not set up. Please check your credentials."
            await assist.TTS(response)

    if "pause" in command:
        result = spot.stop_music()
        if result and "not configured" in result:
            response = "Sorry, Spotify is not set up. Please check your credentials."
            await assist.TTS(response)
    
    if "skip" in command:
        result = spot.skip_to_next()
        if result and "not configured" in result:
            response = "Sorry, Spotify is not set up. Please check your credentials."
            await assist.TTS(response)
    
    if "previous" in command:
        result = spot.skip_to_previous()
        if result and "not configured" in result:
            response = "Sorry, Spotify is not set up. Please check your credentials."
            await assist.TTS(response)
    
    if "spotify" in command:
        spotify_info = spot.get_current_playing_info()
        if spotify_info == "Spotify not configured":
            response = "Sorry, Spotify is not set up. Please check your credentials."
            await assist.TTS(response)
        else:
            query = "System information: " + str(spotify_info)
            print(query)
            response = assist.ask_question_memory(query)
            done = assist.TTS(response)
        

    

        