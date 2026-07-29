from datasets import load_dataset
from pprint import pprint

ds = load_dataset(
    "talkpl-ai/TalkPlayData-Extra-Track-Metadata",
    split="train",
    streaming=True,
)

target_song = "my eyes"
target_artist = "travis scott"

def has_text(values, text):
    if values is None:
        return False
    if not isinstance(values, list):
        values = [values]
    return any(text in str(value).casefold() for value in values)

found = False

for row in ds:
    if has_text(row["track_name"], target_song) and has_text(row["artist_name"], target_artist):
        pprint(row)
        found = True


if not found:
    print("No match found.")