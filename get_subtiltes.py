from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

def get_video_id(url):
    if 'youtu.be/' in url:
        return url.split('youtu.be/')[1].split('?')[0]
    elif 'youtube.com/watch?v=' in url:
        return url.split('youtube.com/watch?v=')[1].split('&')[0]
    else:
        raise ValueError("Invalid YouTube URL format.")
    

def get_subtitles(url, language =['vi', 'en']):
    video_id = get_video_id(url)
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            transcript = transcript_list.find_transcript(language)
        except NoTranscriptFound:
            transcript = transcript_list.find_generated_transcript(language)

        fetched_transcript = transcript.fetch()

        formatter = TextFormatter()
        text = formatter.format_transcript(fetched_transcript)
        print(type(text))
        
        return text
    except ValueError as e:
        print(f"Error: {e}")
    except TranscriptsDisabled:
        print("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        print("No transcript found for this video.")
    except VideoUnavailable:
        print("The video is unavailable.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    url = input("Enter YouTube video URL: ")
    subtitles = get_subtitles(url)
    if subtitles:
        print(subtitles)
    else:
        print("No subtitles available.")