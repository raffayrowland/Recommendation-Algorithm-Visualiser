from database import get_random_songs, get_cfbpr, get_clap
from calculate_combined_vector import calculate_combined_vector

songs = get_random_songs(100)

for song in songs:
    cfbpr = get_cfbpr(song[0])
    clap = get_clap(song[0])

    combined = calculate_combined_vector(cfbpr, clap, 0.5)

    print(combined.shape)