# Spotify Playlist Exporter
<img width="880" height="593" alt="Screenshot 2025-11-19 at 9 16 16 PM" src="https://github.com/user-attachments/assets/057100b9-ddba-4251-aee2-a10ea1c808fe" />

The Spotify Playlist Exporter is a Python project designed to automate the fetching, storage, and export of Spotify playlist data for analysis, backup, or migration. It seamlessly integrates with the Spotify Web API, collects essential song metadata and track IDs, stores everything in a SQLite database, and exports your data to CSV files for use in data analysis tools or personal archives.

## Features

- **API Integration:** Secure authentication and robust interaction with the Spotify Web API to fetch playlist items and search for track IDs.
- **Automated Data Fetching:** Handles playlists of any size by automating offset/limit pagination logic.
- **Database Storage:** Saves playlist tracks and their audio features locally in a SQLite database for persistent, queryable storage.
- **CSV Export:** Exports all playlist data to CSV files (`songs.csv`, `audio_features.csv`) for easy sharing, archiving, or import into tools like Excel and Tableau.
- **Portfolio-Ready:** Demonstrates API integration, data persistence, and Python scripting skills.

## Use Cases

- **Music Enthusiasts:** Back up your favorite playlists for safekeeping.
- **Analysts:** Analyze listening trends and playlist composition.
- **Migrators:** Move playlist data between Spotify accounts or other services.

## Setup

### Prerequisites

- Python 3.x
- [Spotify Developer account](https://developer.spotify.com/)
- Spotify API Client ID and Client Secret

### Install Dependencies

```bash
pip install python-dotenv requests
```

### Project Files

- `main.py`: Authenticates with Spotify, fetches playlist data, and stores it in SQLite.
- `export_spotify_playlist.py`: Exports the database contents to CSV files.
- `songs.csv`, `audio_features.csv`: Example output files.
- `.env`: Store your Spotify API credentials here.

### .env File Example

```
client_id=YOUR_SPOTIFY_CLIENT_ID
client_secret=YOUR_SPOTIFY_CLIENT_SECRET
```

## Usage

### 1. Fetch and Store Playlist Data

Edit `main.py` as needed (set your target playlist ID), then run:

```bash
python main.py
```

Your playlist data will be stored in `spotify_playlist.db`.

### 2. Export Data to CSV

Run the exporter script:

```bash
python export_spotify_playlist.py
```

This will generate `songs.csv` and `audio_features.csv` from the database.

## Example Output

See the included `songs.csv` for the structure and sample data exported from a playlist.

## Customization

- Change the playlist ID in `main.py` to fetch different playlists.
- Modify database schema or CSV export logic to suit your analysis needs.

## Skills Demonstrated

- Third-party API integration (Spotify)
- Secure token-based authentication
- Automated data workflows (pagination, batch processing)
- Data persistence with SQLite
- Data export for analysis and reporting

## License

MIT License

---

**Ideal for**: Music enthusiasts, data analysts, and portfolio projects showing Python scripting, API integration, and data engineering.
