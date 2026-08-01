from database import search_for_song_by_id, search_for_song_by_name, get_nearest_neighbours

song_name = input("Enter song name: ")
search_results = search_for_song_by_name(song_name)

for result in search_results:
    print(result)

song_id = input("ID of the song to query:  ")
neighbours = get_nearest_neighbours(song_id, 0.75)  # 0 means 0% based on cfbpr

for neighbour in neighbours:
    print(search_for_song_by_id(neighbour[0]))