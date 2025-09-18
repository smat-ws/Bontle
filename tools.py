
import asyncio
import assist
# from icrawler.builtin import GoogleImageCrawler
import os
import spot

async def parse_command(command):

    if "play" in command:
        spot.start_music()

    if "pause" in command:
        spot.stop_music()
    
    if "skip" in command:
        spot.skip_to_next()
    
    if "previous" in command:
        spot.skip_to_previous()
    
    if "spotify" in command:
        spotify_info = spot.get_current_playing_info()
        query = "System information: " + str(spotify_info)
        print(query)
        response = assist.ask_question_memory(query)
        done = assist.TTS(response)
        

    

        