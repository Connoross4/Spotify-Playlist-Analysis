import sqlite3
import csv

def export_to_csv(db_file):
    """Export data from the SQLite database to CSV files."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Export songs table
    cursor.execute("SELECT * FROM songs")
    with open('songs.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([i[0] for i in cursor.description])  # Write headers
        writer.writerows(cursor.fetchall())

    # Export audio_features table
    cursor.execute("SELECT * FROM audio_features")
    with open('audio_features.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([i[0] for i in cursor.description])  # Write headers
        writer.writerows(cursor.fetchall())

    conn.close()
    print("Data exported to songs.csv and audio_features.csv")

if __name__ == "__main__":
    export_to_csv('spotify_playlist.db')