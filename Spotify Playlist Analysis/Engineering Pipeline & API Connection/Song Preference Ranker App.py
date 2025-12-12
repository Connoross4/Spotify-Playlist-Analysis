import tkinter as tk
from tkinter import messagebox, filedialog
import sqlite3
import random
import csv
from datetime import datetime
import os
from typing import List, Dict, Tuple
from typing import Tuple, Optional

class SongPreferenceApp:
    def __init__(self, master: tk.Tk):
        self.master: tk.Tk = master
        master.title("Song Preference Ranker")

        self.songs: List[str] = self.load_songs_from_db()
        if not self.songs:
            messagebox.showerror("Error", "No songs found in the database.")
            master.destroy()
            return

        self.ratings: Dict[str, float] = {song: 1400.0 for song in self.songs}
        self.comparison_counts: Dict[str, int] = {song: 0 for song in self.songs}
        self.total_comparisons: int = 0
        self.max_comparisons: int = min(len(self.songs) * 5, 500)  # Limit total comparisons
        self.tournament_rounds: List[List[str]] = [self.songs]
        self.current_round: int = 0

        self.song1: Optional[str] = None
        self.song2: Optional[str] = None

        self.song1_button: tk.Button = tk.Button(master, text="", command=lambda: self.choose_song(0), wraplength=250, height=3)
        self.song1_button.pack(pady=10)

        self.song2_button: tk.Button = tk.Button(master, text="", command=lambda: self.choose_song(1), wraplength=250, height=3)
        self.song2_button.pack(pady=10)

        self.skip_button: tk.Button = tk.Button(master, text="Skip", command=self.skip_comparison)
        self.skip_button.pack(pady=10)

        self.export_button: tk.Button = tk.Button(master, text="Export Rankings", command=self.export_rankings)
        self.export_button.pack(pady=10)

        self.progress_label: tk.Label = tk.Label(master, text="")
        self.progress_label.pack(pady=10)

        self.update_buttons()

    def load_songs_from_db(self) -> List[str]:
        try:
            conn = sqlite3.connect('spotify_playlist.db')
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM songs")
            songs = [row[0] for row in cursor.fetchall()]
            conn.close()
            return songs
        except sqlite3.Error as e:
            messagebox.showerror("Database Error", f"An error occurred: {e}")
            return []

    def get_songs_to_compare(self) -> Tuple[Optional[str], Optional[str]]:
        current_round = self.tournament_rounds[self.current_round]
        if len(current_round) < 2:
            self.current_round += 1
            if self.current_round >= len(self.tournament_rounds):
                return None, None
            current_round = self.tournament_rounds[self.current_round]
        
        song1, song2 = random.sample(current_round, 2)
        return song1, song2

    def update_buttons(self) -> None:
        if self.total_comparisons >= self.max_comparisons or self.is_ranking_stable():
            self.show_results()
            return

        self.song1, self.song2 = self.get_songs_to_compare()
        if self.song1 is None or self.song2 is None:
            self.show_results()
            return

        self.song1_button.config(text=self.song1)
        self.song2_button.config(text=self.song2)
        self.progress_label.config(text=f"Round {self.current_round + 1}, Comparison {self.total_comparisons + 1} of max {self.max_comparisons}")

    def choose_song(self, choice: int) -> None:
        winner = self.song1 if choice == 0 else self.song2
        loser = self.song2 if choice == 0 else self.song1

        if winner is None or loser is None:
            messagebox.showerror("Error", "Winner or loser is None.")
            return

        self.update_elo(winner, loser)
        self.comparison_counts[winner] += 1
        self.comparison_counts[loser] += 1
        self.total_comparisons += 1

        current_round = self.tournament_rounds[self.current_round]
        if loser in current_round:
            current_round.remove(loser)
        
        if len(current_round) == 1 and self.current_round == len(self.tournament_rounds) - 1:
            self.tournament_rounds.append(current_round[:])

        self.update_buttons()

    def update_elo(self, winner: str, loser: str) -> None:
        k: float = 32.0
        r1: float = self.ratings[winner]
        r2: float = self.ratings[loser]

        e1: float = 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))
        e2: float = 1.0 / (1.0 + 10.0 ** ((r1 - r2) / 400.0))

        self.ratings[winner] = r1 + k * (1.0 - e1)
        self.ratings[loser] = r2 + k * (0.0 - e2)

    def skip_comparison(self) -> None:
        self.total_comparisons += 1
        self.update_buttons()

    def is_ranking_stable(self) -> bool:
        if self.total_comparisons < len(self.songs):
            return False
        
        sorted_ratings = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)
        top_10_percent = sorted_ratings[:max(3, len(sorted_ratings) // 10)]
        return all(self.comparison_counts[song] >= 3 for song, _ in top_10_percent)

    def get_ranked_songs(self) -> List[str]:
        return sorted(self.songs, key=lambda s: self.ratings[s], reverse=True)

    def export_rankings(self) -> None:
        ranked_songs = self.get_ranked_songs()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"song_rankings_{timestamp}.csv"
        )
        
        if filename:
            if os.path.exists(filename):
                os.remove(filename)
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Rank", "Song", "Rating"])
                for rank, song in enumerate(ranked_songs, 1):
                    writer.writerow([rank, song, round(self.ratings[song], 2)])
            
            messagebox.showinfo("Export Successful", f"Rankings exported to {filename}")

    def show_results(self) -> None:
        ranked_songs = self.get_ranked_songs()

        result_window = tk.Toplevel(self.master)
        result_window.title("Ranked Songs")
        
        listbox = tk.Listbox(result_window, width=50)
        listbox.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        for i, song in enumerate(ranked_songs, 1):
            listbox.insert(tk.END, f"{i}. {song} (Rating: {round(self.ratings[song], 2)})")

        export_button = tk.Button(result_window, text="Export Final Rankings", 
                                  command=self.export_rankings)
        export_button.pack(pady=10)

        self.master.destroy()  # Close the main window

def main():
    root = tk.Tk()
    app = SongPreferenceApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()