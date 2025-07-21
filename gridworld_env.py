import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import defaultdict
import time 
import pickle
import os

# Try to import rendering from various locations
try:
    from gymnasium.envs.classic_control import rendering
except ImportError:
    try:
        from gymnasium.envs.classic_control.utils import rendering
    except ImportError:
        # If rendering is not available, create a minimal implementation
        import pygame
        import math
        
        class SimpleRenderer:
            def __init__(self, width, height):
                pygame.init()
                self.width = width
                self.height = height
                self.screen = pygame.display.set_mode((width, height))
                pygame.display.set_caption("GridWorld")
                self.clock = pygame.time.Clock()
                self.objects = []
                
            def add_geom(self, obj):
                self.objects.append(obj)
                
            def render(self, return_rgb_array=False):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                
                self.screen.fill((255, 255, 255))  # White background
                
                for obj in self.objects:
                    obj.render(self.screen)
                
                if return_rgb_array:
                    array = pygame.surfarray.array3d(self.screen)
                    array = np.transpose(array, (1, 0, 2))
                    return array
                else:
                    pygame.display.flip()
                    return None
                    
            def close(self):
                pygame.quit()
        
        class SimpleGeom:
            def __init__(self):
                self.attrs = []
                
            def add_attr(self, attr):
                self.attrs.append(attr)
                
            def render(self, surface):
                pass
        
        class SimpleTransform:
            def __init__(self):
                self.translation = (0, 0)
                
            def set_translation(self, x, y):
                self.translation = (x, y)
        
        class SimpleCircle(SimpleGeom):
            def __init__(self, radius):
                super().__init__()
                self.radius = radius
                self.color = (0, 0, 0)
                
            def set_color(self, r, g, b):
                self.color = (int(r*255), int(g*255), int(b*255))
                
            def render(self, surface):
                pos = (0, 0)
                for attr in self.attrs:
                    if hasattr(attr, 'translation'):
                        pos = attr.translation
                pygame.draw.circle(surface, self.color, (int(pos[0]), int(pos[1])), int(self.radius))
        
        class SimplePolygon(SimpleGeom):
            def __init__(self, points):
                super().__init__()
                self.points = points
                self.color = (0, 0, 0)
                
            def set_color(self, r, g, b):
                self.color = (int(r*255), int(g*255), int(b*255))
                
            def render(self, surface):
                if len(self.points) >= 3:
                    pygame.draw.polygon(surface, self.color, self.points)
        
        class SimplePolyLine(SimpleGeom):
            def __init__(self, points, closed=False):
                super().__init__()
                self.points = points
                self.closed = closed
                self.color = (0, 0, 0)
                self.linewidth = 1
                
            def set_color(self, r, g, b):
                self.color = (int(r*255), int(g*255), int(b*255))
                
            def set_linewidth(self, width):
                self.linewidth = width
                
            def render(self, surface):
                if len(self.points) >= 2:
                    pygame.draw.lines(surface, self.color, self.closed, self.points, self.linewidth)
        
        # Create a simple rendering module
        class rendering:
            @staticmethod
            def Viewer(width, height):
                return SimpleRenderer(width, height)
                
            @staticmethod
            def make_circle(radius):
                return SimpleCircle(radius)
                
            @staticmethod
            def FilledPolygon(points):
                return SimplePolygon(points)
                
            @staticmethod
            def PolyLine(points, closed=False):
                return SimplePolyLine(points, closed)
                
            @staticmethod
            def Transform():
                return SimpleTransform()

CELL_SIZE = 100
MARGIN = 10

def get_coords(row, col, loc='center'):
    xc = (col + 1.5) * CELL_SIZE
    yc = (row + 1.5) * CELL_SIZE
    if loc == 'center':
        return xc, yc
    elif loc == 'interior_corners':
        half_size = CELL_SIZE//2 - MARGIN
        xl, xr = xc - half_size, xc + half_size
        yt, yb = yc - half_size, yc + half_size
        return [(xl, yt), (xr, yt), (xr, yb), (xl, yb)]
    elif loc == 'interior_triangle':
        x1, y1 = xc, yc + CELL_SIZE//3
        x2, y2 = xc + CELL_SIZE//3, yc - CELL_SIZE//3
        x3, y3 = xc - CELL_SIZE//3, yc - CELL_SIZE//3
        return [(x1, y1), (x2, y2), (x3, y3)]
    
def draw_object(coord_list):
    if len(coord_list) == 1: # -> circle
        obj = rendering.make_circle(int(0.45* CELL_SIZE))
        obj_transform = rendering.Transform()
        obj.add_attr(obj_transform)
        obj_transform.set_translation(*coord_list[0])
        obj.set_color(0.2, 0.2, 0.2) # -> black
    elif len(coord_list) == 3: # -> triangle
        obj = rendering.FilledPolygon(coord_list)
        obj.set_color(0.9, 0.6, 0.2) # -> yellow
    elif len(coord_list) > 3: # -> polygon
        obj = rendering.FilledPolygon(coord_list)
        obj.set_color(0.4, 0.4, 0.8) # -> blue
    return obj

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, num_rows=4, num_cols=6, delay=0.05, render_mode=None):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.delay = delay
        self.render_mode = render_mode
        
        # Define movement functions
        move_up = lambda row, col: (max(row-1, 0), col)
        move_down = lambda row, col: (min(row+1, num_rows-1), col)
        move_left = lambda row, col: (row, max(col-1, 0))
        move_right = lambda row, col: (row, min(col+1, num_cols-1))
        self.action_defs = {
            0: move_up,
            1: move_down,
            2: move_left,
            3: move_right
        }
        
        # Number of states/actions
        nS = num_cols * num_rows
        nA = len(self.action_defs)
        
        # Define observation and action spaces
        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)
        
        # State conversion dictionaries
        self.grid2state_dict = {(s//num_cols, s%num_cols):s for s in range(nS)}
        self.state2grid_dict = {s:(s//num_cols, s%num_cols) for s in range(nS)}
        
        # Gold state 
        gold_cell = (num_rows//2, num_cols-2)
        
        # Trap states 
        trap_cells = [((gold_cell[0] + 1), gold_cell[1]), 
                     (gold_cell[0], gold_cell[1]-1), 
                     ((gold_cell[0] -1), gold_cell[1])]
        
        gold_state = self.grid2state_dict[gold_cell]
        trap_states = [self.grid2state_dict[(r,c)] for (r,c) in trap_cells]
        self.terminal_states = [gold_state] + trap_states
        self.gold_state = gold_state
        
        print(f"Terminal states: {self.terminal_states}")
        print(f"Gold state: {self.gold_state}")
        
        # Build the transition probability
        self.P = defaultdict(dict)
        for s in range(nS):
            row, col = self.state2grid_dict[s]
            self.P[s] = defaultdict(list)
            for a in range(nA):
                action = self.action_defs[a]
                next_s = self.grid2state_dict[action(row, col)]
                
                # Terminal state rewards
                if self.is_terminal(next_s):
                    r = (1.0 if next_s == self.gold_state else -1.0)
                else:
                    r = 0.0
                
                # Check if current state is terminal
                if self.is_terminal(s):
                    done = True
                    next_s = s
                else:
                    done = False
                
                self.P[s][a] = [(1.0, next_s, r, done)]
        
        # Initialize state
        self.s = 0  # Start at state 0
        self.viewer = None
        self.gold_cell = gold_cell
        self.trap_cells = trap_cells

    def is_terminal(self, state):
        return state in self.terminal_states
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.s = 0  # Always start at state 0
        if self.render_mode == "human":
            self._build_display()
        return self.s, {}
    
    def step(self, action):
        transitions = self.P[self.s][action]
        i = 0  # Since we only have one transition per state-action pair
        p, next_s, r, done = transitions[i]
        self.s = next_s
        
        # Return observation, reward, terminated, truncated, info
        return next_s, r, done, False, {}
    
    def _build_display(self):
        if self.viewer is not None:
            return
            
        screen_width = (self.num_cols+2) * CELL_SIZE
        screen_height = (self.num_rows+2) * CELL_SIZE
        self.viewer = rendering.Viewer(screen_width, screen_height)
        all_objects = []
        
        # Border
        bp_list = [(CELL_SIZE-MARGIN, CELL_SIZE-MARGIN), 
                   (screen_width-CELL_SIZE+MARGIN, CELL_SIZE-MARGIN),
                   (screen_width-CELL_SIZE + MARGIN, screen_height-CELL_SIZE + MARGIN), 
                   (CELL_SIZE-MARGIN, screen_height-CELL_SIZE+MARGIN)]
        border = rendering.PolyLine(bp_list, True)
        if hasattr(border, 'set_linewidth'):
            border.set_linewidth(5)
        all_objects.append(border)
        
        # Vertical lines
        for col in range(self.num_cols + 1):
            x1, y1 = (col + 1) * CELL_SIZE, CELL_SIZE
            x2, y2 = (col + 1) * CELL_SIZE, (self.num_rows+1)*CELL_SIZE
            line = rendering.PolyLine([(x1, y1), (x2, y2)], False)
            all_objects.append(line)
        
        # Horizontal lines 
        for row in range(self.num_rows + 1):
            x1, y1 = CELL_SIZE, (row + 1) * CELL_SIZE
            x2, y2 = (self.num_cols + 1) * CELL_SIZE, (row + 1) * CELL_SIZE
            line = rendering.PolyLine([(x1, y1), (x2, y2)], False)
            all_objects.append(line)

        # Traps --> circles
        for cell in self.trap_cells:
            trap_coords = get_coords(*cell, loc='center')
            all_objects.append(draw_object([trap_coords]))

        # Gold --> triangle
        gold_coords = get_coords(*self.gold_cell, loc='interior_triangle')
        all_objects.append(draw_object(gold_coords))
        
        # Agent --> square 
        if (os.path.exists('robot-coordinates.pkl') and CELL_SIZE == 100):
            agent_coords = pickle.load(open('robot-coordinates.pkl', 'rb'))
            starting_coords = get_coords(0, 0, loc='center')
            agent_coords += np.array(starting_coords)
        else:
            agent_coords = get_coords(0, 0, loc='interior_corners')
        
        agent = draw_object(agent_coords)
        self.agent_trans = rendering.Transform()
        agent.add_attr(self.agent_trans)
        all_objects.append(agent)
        
        for obj in all_objects:
            self.viewer.add_geom(obj)

    def render(self):
        if self.render_mode is None:
            return
            
        if self.viewer is None:
            self._build_display()
            
        if self.viewer is None:
            return
            
        # Calculate agent position
        x_coord = self.s % self.num_cols
        y_coord = self.s // self.num_cols
        x_coord = x_coord * CELL_SIZE
        y_coord = y_coord * CELL_SIZE
        self.agent_trans.set_translation(x_coord, y_coord)
        
        # Render based on mode
        if self.render_mode == "human":
            rend = self.viewer.render(return_rgb_array=False)
            time.sleep(self.delay)
            return rend
        elif self.render_mode == "rgb_array":
            return self.viewer.render(return_rgb_array=True)
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

if __name__ == "__main__":
    env = GridWorldEnv(5, 6, render_mode="human")
    for i in range(1):
        obs, info = env.reset()
        env.render()
        while True:
            action = np.random.choice(env.action_space.n)
            obs, reward, terminated, truncated, info = env.step(action)
            print(f'Action {action} -> State: {obs}, Reward: {reward}, Done: {terminated}')
            env.render()
            if terminated or truncated:
                break
    
    env.close()
