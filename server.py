import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from build_vector_space import build_vector_space, add_additional_points
from database import search_for_song_by_name, get_nearest_neighbours
import json
import requests

app = FastAPI()
app.mount("/frontend", StaticFiles(directory="frontend"))

@app.get("/")  # Returns the main HTML file
def get_home():
    return FileResponse(f"frontend/index.html")

@app.get("/api/space")  # Fetches / computes JSON of the 3D space for given parameters
def get_songs(n, alpha):
    n = int(n)

    if not 100 <= n <= 20_000:
        raise HTTPException(
            status_code=400,
            detail="Number of songs must be between 100 and 20,000"
        )

    if alpha not in ["000", "025", "050", "075", "100"]:
        raise HTTPException(
            status_code=400,
            detail="Alpha must be 0, 0.25, 0.5, 0.75, or 1"
        )

    songs_and_embeddings = build_vector_space(n, alpha)
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
def get_true_nearest_neighbours(track_id, n, alpha):
    n = int(n)
    if not 100 <= n <= 20_000:
        raise HTTPException(
            status_code=400,
            detail="Number of songs must be between 100 and 20,000"
        )

    if alpha not in ["000", "025", "050", "075", "100"]:
        raise HTTPException(
            status_code=400,
            detail="Alpha must be 0, 0.25, 0.5, 0.75, or 1"
        )

    neighbours = get_nearest_neighbours(track_id, alpha)
    if not neighbours:
        raise HTTPException(status_code=404, detail="Track not found")

    existing_space = build_vector_space(n, alpha)
    neighbour_points = add_additional_points(
        neighbours, existing_space, n, alpha
    )

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
        "picture_link": response["contributors"][0]["picture_medium"]
    }

    return data


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
