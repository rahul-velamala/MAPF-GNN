#!/usr/bin/env python3
import yaml
import matplotlib

# matplotlib.use("Agg")
from matplotlib.patches import Circle, Rectangle, Arrow, PathPatch
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import matplotlib.animation as manimation
import argparse
import math
from matplotlib.path import Path
from matplotlib import cm

# More visually appealing color palette
AGENT_COLOR = "#3498db"  # Bright blue
GOAL_COLOR = "#2ecc71"   # Bright green
OBSTACLE_COLOR = "#e74c3c"  # Bright red
COLLISION_COLOR = "#e67e22"  # Orange
BACKGROUND_COLOR = "#f8f9fa"  # Light gray
BORDER_COLOR = "#2c3e50"  # Dark blue
TEXT_COLOR = "#ffffff"   # White

class Animation:
    def __init__(self, map, schedule):
        self.map = map
        self.schedule = schedule
        self.combined_schedule = {}
        self.combined_schedule.update(self.schedule["schedule"])
        print(self.combined_schedule)

        aspect = map["map"]["dimensions"][0] / map["map"]["dimensions"][1]

        # Set up a more visually appealing figure
        self.fig = plt.figure(figsize=(4 * aspect, 4), facecolor=BACKGROUND_COLOR)
        self.ax = self.fig.add_subplot(111, aspect="equal")
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)
        
        # Set background color
        self.ax.set_facecolor(BACKGROUND_COLOR)
        
        # Remove axis ticks for cleaner look
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.patches = []
        self.artists = []
        self.agents = dict()
        self.agent_names = dict()
        
        # Create boundary patch
        xmin = -0.5
        ymin = -0.5
        xmax = map["map"]["dimensions"][0] - 0.5
        ymax = map["map"]["dimensions"][1] - 0.5

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        # Add grid for better visual reference
        self.ax.grid(True, color='#ddd', linestyle='-', linewidth=0.5, alpha=0.7)
        
        # Create border with better styling
        self.patches.append(
            Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                facecolor="none",
                edgecolor=BORDER_COLOR,
                linewidth=2.0
            )
        )
        
        # Make obstacles more visually distinct
        for o in map["map"]["obstacles"]:
            x, y = o[0], o[1]
            # Add shadow effect for obstacles
            shadow = Rectangle(
                (x - 0.45, y - 0.45),
                0.9,
                0.9,
                facecolor='black',
                alpha=0.2
            )
            self.patches.append(shadow)
            
            obstacle = Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                facecolor=OBSTACLE_COLOR,
                edgecolor="#c0392b",  # Darker red for edge
                linewidth=1.5,
                alpha=0.8
            )
            self.patches.append(obstacle)

        # Create agents
        self.T = 0
        
        # Draw goals with clearer visual styling
        for d, i in zip(map["agents"], range(0, len(map["agents"]))):
            # Add subtle glow/highlight for goals
            glow = Rectangle(
                (d["goal"][0] - 0.3, d["goal"][1] - 0.3),
                0.6,
                0.6,
                facecolor=GOAL_COLOR,
                alpha=0.3
            )
            self.patches.append(glow)
            
            goal = Rectangle(
                (d["goal"][0] - 0.25, d["goal"][1] - 0.25),
                0.5,
                0.5,
                facecolor=GOAL_COLOR,
                edgecolor="#27ae60",  # Darker green for edge
                linewidth=1.5,
                alpha=0.8,
                zorder=1
            )
            self.patches.append(goal)
            
        # Create agent circles with improved styling
        for d, i in zip(map["agents"], range(0, len(map["agents"]))):
            name = d["name"]
            # Add shadow effect
            shadow = Circle(
                (d["start"][0] + 0.05, d["start"][1] - 0.05),
                0.32,
                facecolor='black',
                alpha=0.2,
                zorder=2
            )
            self.patches.append(shadow)
            
            # Create agent with gradient effect
            self.agents[name] = Circle(
                (d["start"][0], d["start"][1]),
                0.3,
                facecolor=AGENT_COLOR,
                edgecolor="#2980b9",  # Darker blue for edge
                linewidth=1.5,
                zorder=3
            )
            self.agents[name].original_face_color = AGENT_COLOR
            self.patches.append(self.agents[name])
            
            self.T = max(self.T, schedule["schedule"][name][-1]["t"])
            
            # Improve text visibility
            self.agent_names[name] = self.ax.text(
                d["start"][0], d["start"][1], name.replace("agent", ""),
                color=TEXT_COLOR,
                fontweight='bold',
                fontsize=9,
                zorder=4
            )
            self.agent_names[name].set_horizontalalignment("center")
            self.agent_names[name].set_verticalalignment("center")
            self.artists.append(self.agent_names[name])

        # Add title with styling
        title = self.ax.text(
            (xmax + xmin) / 2, ymax - 0.3, 
            "Agent Path Visualization",
            fontsize=14,
            fontweight='bold',
            ha='center',
            color=BORDER_COLOR
        )
        self.artists.append(title)

        self.anim = animation.FuncAnimation(
            self.fig,
            self.animate_func,
            init_func=self.init_func,
            frames=int(self.T + 1) * 10,
            interval=100,
            blit=True,
        )

    def save(self, file_name, speed):
        self.anim.save(
            file_name, 
            "ffmpeg", 
            fps=10 * speed, 
            dpi=200,
            savefig_kwargs={"facecolor": self.fig.get_facecolor()}
        )

    def show(self):
        plt.show()

    def init_func(self):
        for p in self.patches:
            self.ax.add_patch(p)
        for a in self.artists:
            self.ax.add_artist(a)
        return self.patches + self.artists

    def animate_func(self, i):
        for agent_name, agent in self.combined_schedule.items():
            pos = self.getState(i / 10, agent)
            p = (pos[0], pos[1])
            self.agents[agent_name].center = p
            self.agent_names[agent_name].set_position(p)

        # Reset all colors
        for _, agent in self.agents.items():
            agent.set_facecolor(agent.original_face_color)

        # Check drive-drive collisions with improved visual feedback
        agents_array = [agent for _, agent in self.agents.items()]
        collisions = False
        for i in range(0, len(agents_array)):
            for j in range(i + 1, len(agents_array)):
                d1 = agents_array[i]
                d2 = agents_array[j]
                pos1 = np.array(d1.center)
                pos2 = np.array(d2.center)
                if np.linalg.norm(pos1 - pos2) < 0.7:
                    d1.set_facecolor(COLLISION_COLOR)
                    d2.set_facecolor(COLLISION_COLOR)
                    d1.set_edgecolor("#d35400")  # Darker orange
                    d2.set_edgecolor("#d35400")
                    collisions = True
                    print("COLLISION! (agent-agent) ({}, {})".format(i, j))
        
        return self.patches + self.artists

    def getState(self, t, d):
        idx = 0
        while idx < len(d) and d[idx]["t"] < t:
            idx += 1
        if idx == 0:
            return np.array([float(d[0]["x"]), float(d[0]["y"])])
        elif idx < len(d):
            posLast = np.array([float(d[idx - 1]["x"]), float(d[idx - 1]["y"])])
            posNext = np.array([float(d[idx]["x"]), float(d[idx]["y"])])
        else:
            return np.array([float(d[-1]["x"]), float(d[-1]["y"])])
        dt = d[idx]["t"] - d[idx - 1]["t"]
        t = (t - d[idx - 1]["t"]) / dt
        pos = (posNext - posLast) * t + posLast
        return pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("map", help="input file containing map")
    parser.add_argument("schedule", help="schedule for agents")
    parser.add_argument(
        "--video",
        dest="video",
        default=None,
        help="output video file (or leave empty to show on screen)",
    )
    parser.add_argument("--speed", type=int, default=1, help="speedup-factor")
    args = parser.parse_args()

    with open(args.map) as map_file:
        map = yaml.load(map_file, Loader=yaml.FullLoader)

    with open(args.schedule) as states_file:
        schedule = yaml.load(states_file, Loader=yaml.FullLoader)

    animation = Animation(map, schedule)

    if args.video:
        animation.save(args.video, args.speed)
    else:
        animation.show()