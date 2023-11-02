from pytube import YouTube
import sys


def download_video(url, outputPath: None):
    try:
        yt = YouTube(url, on_progress_callback=progress_callback)

        video_stream = yt.streams.filter(res='1080p',progressive=True, file_extension='mp4').first() or yt.streams.get_highest_resolution()

        if outputPath:
            video_stream.download(outputPath)
            print(f'Downloaded video to {outputPath}')
        else:
            video_stream.download()
            print("Downloaded video")


    except Exception as e:
        print(f'error downloading video: {e}')

def progress_callback(stream, chunk, bytes_remaining):
    # Calculate the download percentage
    file_size = stream.filesize
    bytes_downloaded = file_size - bytes_remaining
    percent_complete = (bytes_downloaded / file_size) * 100
    sys.stdout.write(f"\rDownloaded: {percent_complete:.2f}%")
    sys.stdout.flush()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_url = sys.argv[1]
        download_video(video_url, 'videos/raw')
    else:
        print("Please provide a video url")