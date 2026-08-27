import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from build_vector_space import build_vector_space, add_additional_points
from database import search_for_song_by_name, get_nearest_neighbours, get_info_single_song
import json
import requests

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI()
app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR))

@app.get("/")  # Returns the main HTML file
def get_home():
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/api/space")  # Fetches / computes JSON of the 3D space for given parameters
def get_songs(n):
    n = int(n)

    if not 100 <= n <= 20_000:
        raise HTTPException(
            status_code=400,
            detail="Number of songs must be between 100 and 20,000"
        )

    songs_and_embeddings = build_vector_space(n)
    data = songs_and_embeddings[
        [
            "track_id",
            "track_name",
            "artist_name",
            "isrc",
            "x",
            "y",
            "z",
        ]
    ]

    return json.loads(data.to_json(orient="records", force_ascii=False))

@app.get("/api/search/")  # Searches for a song by name or artist
def search_song_by_name(query, n):
    songs = search_for_song_by_name(query, n)
    songs_json = [
        {
            "track_id": track_id,
            "track_name": track_name,
            "artist_name": artist_names
        } for track_id, track_name, artist_names in songs
    ]

    return {"songs": songs_json}

@app.get("/api/nn")  # Gets the true nearest neighbours of a song from a track id
def get_true_nearest_neighbours(track_id, n):
    n = int(n)
    if not 100 <= n <= 20_000:
        raise HTTPException(
            status_code=400,
            detail="Number of songs must be between 100 and 20,000"
        )

    neighbours = get_nearest_neighbours(track_id)
    if not neighbours:
        raise HTTPException(status_code=404, detail="Track not found")

    existing_space = build_vector_space(n)
    points_to_display = neighbours

    if track_id not in set(existing_space["track_id"]):
        searched_track = get_info_single_song(track_id)
        if searched_track is None:
            raise HTTPException(status_code=404, detail="Track not found")
        points_to_display = [searched_track, *neighbours]

    neighbour_points = add_additional_points(points_to_display, existing_space, n)

    data = neighbour_points[
        [
            "track_id",
            "track_name",
            "artist_name",
            "isrc",
            "x",
            "y",
            "z",
        ]
    ]

    return json.loads(data.to_json(orient="records", force_ascii=False))

@app.get("/api/preview")  # Gets the audio preview and cover art for a given ISRC
def get_preview(isrc):
    response = requests.get(f"https://api.deezer.com/track/isrc:{isrc}")
    response = response.json()

    if "error" in response:
        raise HTTPException(status_code=404, detail="Track not found")

    data = {
        "title": response["title"],
        "artist": response["artist"]["name"],
        "preview_link": response["preview"],
        "picture_link": response["album"]["cover_medium"]
    }

    return data


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
