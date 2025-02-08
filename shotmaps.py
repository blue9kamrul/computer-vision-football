import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch

# Load the CSV file (update the filename as needed)
file_path = 'csv/shotmaps.csv'  # Replace with your CSV file
shots_df = pd.read_csv(file_path)

# Create a half-pitch view
pitch = Pitch(pitch_type='statsbomb', half=True, line_color='black')
fig, ax = pitch.draw(figsize=(10, 7))

# Scatter plot of shots
shot_colors = []
for _, shot in shots_df.iterrows():
    x, y = shot['x'], shot['y']
    minute, second = shot['minute'], shot['second']
    on_target = bool(shot['outcome'])  # Ensure it's a boolean
    
    color = 'red' if on_target else 'blue'
    shot_colors.append(color)
    ax.scatter(x, y, s=100, color=color, edgecolors='black')
    ax.text(x, y, f'{minute}:{second}', fontsize=8, ha='right', color='white', 
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

# Add legend
handles = [plt.Line2D([0], [0], marker='o', color='red', markersize=10, linestyle='None', label='On Target'),
           plt.Line2D([0], [0], marker='o', color='blue', markersize=10, linestyle='None', label='Off Target')]
ax.legend(handles=handles, loc='upper right')

plt.title('Shot Map (Half-Pitch)')
plt.show()