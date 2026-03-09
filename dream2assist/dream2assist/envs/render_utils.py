import functools
import math
from pandas import read_csv
import pathlib
import torch
from typing import Optional, Union

import numpy as np

import gymnasium as gym
from gymnasium import spaces
import gymnasium.envs.box2d.car_dynamics as car_dynamics
from gymnasium.envs.box2d.car_dynamics import Car
from gymnasium.error import DependencyNotInstalled, InvalidAction
from gymnasium.utils import EzPickle


try:
    import Box2D
    from Box2D.b2 import contactListener, fixtureDef, polygonShape
except ImportError as e:
    raise DependencyNotInstalled("Box2D is not installed, run `pip install gymnasium[box2d]`") from e

try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
    from pygame import gfxdraw
except ImportError as e:
    raise DependencyNotInstalled("pygame is not installed, run `pip install gymnasium[box2d]`") from e


VID_SCALE = 2
STATE_W = 64  # less than Atari 160x192
STATE_H = 64
VIDEO_W = 600 // VID_SCALE
VIDEO_H = 400 // VID_SCALE
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 6000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 1.0  # Camera zoom
ZOOM_FOLLOW = False  # Set to False for fixed view (don't use zoom)

TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
GRASS_DIM = PLAYFIELD / 20.0
MAX_SHAPE_DIM = max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE

ADO_STATE_OBS_LOOKAHEAD = 20
ADO_STATE_OBS_SHIFT = 2  # Offset of tiles in state observation. 0 = starting with the tile closest to the car
STATE_OBS_LOOKAHEAD = 10  # Number of tiles to look ahead in the state observation
STATE_OBS_SHIFT = 4  # Offset of tiles in state observation. 0 = starting with the tile closest to the car

# Specify different car colors
CAR_COLORS = [
    (0.8, 0.0, 0.0),
    (0.0, 0.0, 0.8),
    (0.0, 0.8, 0.0),
    (0.0, 0.8, 0.8),
    (0.8, 0.8, 0.8),
    (0.0, 0.0, 0.0),
    (0.8, 0.0, 0.8),
    (0.8, 0.8, 0.0),
]


class FrictionDetector(contactListener):
    def __init__(self, env, lap_complete_percent):
        contactListener.__init__(self)
        self.env = env
        self.lap_complete_percent = lap_complete_percent

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        # inherit tile color from env
        tile.color[:] = self.env.road_color
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited[obj.car_id]:
                tile.road_visited[obj.car_id] = True
                self.env.tile_visited_count[obj.car_id] += 1

                # The reward is dampened on tiles that have been visited already.
                past_visitors = sum(tile.road_visited) - 1
                reward_factor = 1 - (past_visitors / self.env.num_agents)
                self.env.reward += reward_factor * 1000.0 / len(self.env.track)

                # Lap is considered completed if enough % of the track was covered
                if (
                    tile.idx == 0
                    and self.env.tile_visited_count[obj.car_id] / len(self.env.track) > self.lap_complete_percent
                ):
                    self.env.new_lap[obj.car_id] = True
        else:
            obj.tiles.remove(tile)


class Render:
    def __init__(
        self,
        env,
        actions,
        states,
        reward,
        reward_components = None,
        domain_randomize: bool = False,
    ):
        self.domain_randomize = domain_randomize
        self._init_colors()
        self.surf = None
        self.screen = None
        self.lap_complete_percent = 0.95
        self.contactListener_keepref = FrictionDetector(self, self.lap_complete_percent)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)

        self.action_names = env.actions
        self.continuous = env.continuous
        self.actions = actions[1:]  # T x N_cars
        self.car_states = states[1:]  # T x N_cars x N_states
        self.rewards = reward[1:]  # T x N_ai_human
        self.reward_components = reward_components if reward_components is not None else None  # T x N_components
        if self.reward_components is not None and isinstance(self.reward_components, np.ndarray):
            # Convert list of dicts to dict of lists for easier indexing during rendering
            keys = env.reward_component_names
            labeled_components = []
            # Skip processing if reward_components is empty
            if self.reward_components.size == 0:
                self.reward_components = None
            elif keys is not None:
                assert self.reward_components.shape[1] == len(keys), (
                    f"Reward components shape mismatch: "
                    f"array has {self.reward_components.shape[1]} components, "
                    f"but env.reward_component_names has {len(keys)} names: {keys}"
                )
                for comp in self.reward_components:
                    labeled_components.append({
                        key: comp[i] for i, key in enumerate(keys)
                    })
            else:
                for comp in self.reward_components:
                    labeled_components.append({
                        f"reward component {i}": comp[i] for i in range(comp.shape[0])
                    })
            self.reward_components = labeled_components

        states = [[float(v) for v in c] for c in self.car_states[0]]
        self.angle_xy_indices = [2, 0, 1]  # angle, x, y
        angle_xy = [[s[i] for i in self.angle_xy_indices] for s in states]
        self.cars = [Car(self.world, *s) for s in angle_xy]
        self.clock = None
        self.road_poly = env.road_poly
        self.has_reset = False

        self.index = 0

    def _init_colors(self):
        if self.domain_randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20
        else:
            # default colours
            self.road_color = np.array([102, 102, 102])
            self.bg_color = np.array([102, 204, 102])
            self.grass_color = np.array([102, 230, 102])

    def _destroy(self):
        for car in self.cars:
            assert car is not None
            car.destroy()

    def render_video(self, mode):
        video = []
        for action, state in zip(self.actions, self.car_states):
            video.append(self.step(action, state))
            self.index += 1
        self._destroy()
        return np.stack(video, axis=0)

    def step(self, actions, states):
        # print(f"   RENDER STEP")
        for cid, (action, state) in enumerate(zip(actions, states)):
            if action is not None:
                self.has_reset = (
                    True  # Currently using action being populated as a means for checking if reset has been called.
                )
                if self.continuous:
                    self.cars[cid].steer(float(-action[0]))
                    self.cars[cid].gas(float(max(action[1], 0.0)))
                    self.cars[cid].brake(float(-min(action[1], 0.0)))
                else:  # TODO Remove this portion and force all consumers to be continuous or convert to it before stepping.
                    noop = "NOOP" in self.action_names[action]
                    left = "LEFT" in self.action_names[action]
                    right = "RIGHT" in self.action_names[action]
                    gas = "GAS" in self.action_names[action]
                    brake = "BRAKE" in self.action_names[action]
                    self.cars[cid].steer(-0.6 * left + 0.6 * right)
                    self.cars[cid].gas(0.2 * gas)
                    self.cars[cid].brake(0.8 * brake)

            self.cars[cid].step(1.0 / FPS)
            # Correct the angle / position to be precisely as recorded.
            self.cars[cid].hull.angle = float(state[self.angle_xy_indices[0]])
            self.cars[cid].hull.position = (float(state[self.angle_xy_indices[1]]), float(state[self.angle_xy_indices[2]]))
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        # print(f" render position {self.cars[0].hull.position[0]}, {self.cars[0].hull.position[1]}")

        # TODO Check that the car states match the trajectory rollouts after stepping from the actions above.

        return self.render("rgb_array")

    def render(self, mode):
        if mode == "train":
            return np.zeros((self.num_agents, VIDEO_H, VIDEO_W, 3))
        num_agents = self.actions.shape[1]
        result = []
        for cur_car_id in range(num_agents):
            result.append(self.render_window(cur_car_id, mode))
        return np.stack(result, axis=0)

    def render_window(self, car_id: int, mode: str):
        car = self.cars[car_id]
        # print(f"   RENDER WINDOW car position {car.hull.position}")
        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if not self.has_reset:
            return  # reset() not called yet

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        assert car is not None
        # computing transformations
        angle = -car.hull.angle

        zoom = ZOOM * SCALE
        scroll_x = -(car.hull.position[0]) * zoom
        scroll_y = -(car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        self.render_road(zoom, trans, angle)
        for id, car in enumerate(self.cars):
            car.hull.color = (0.0, 0.0, 0.8)  # Set all other car colors to blue
            if id == car_id:  # Ego car
                car.hull.color = (0.8, 0.0, 0.0)  # Set ego car color to red
            car.draw(
                self.surf,
                zoom,
                trans,
                angle,
                mode not in ["state_pixels_list", "state_pixels"],
            )
        self.surf = pygame.transform.flip(self.surf, False, True)

        # showing stats
        self.render_indicators(car, WINDOW_W, WINDOW_H)

        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render("%04i" % self.rewards[self.index][car_id], True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (150, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        self.surf.blit(text, text_rect)

        if self.reward_components is not None:
            font = pygame.font.Font(pygame.font.get_default_font(), 40)
            color = (0, 0, 0)
            y_offset = -19 * 40  # Start text roughly 19 lines from the bottom of the image.
            for component, value in self.reward_components[self.index].items():
                component_text = font.render(f'{component}: {value:.2f}', True, color)
                self.surf.blit(component_text, (100, WINDOW_H - WINDOW_H * 2.5 / 40.0 + y_offset))
                y_offset += 40

        # ego_pos = self.cars[car_id].hull.position
        # ado_pos = self.ado_car.hull.position
        # ego_ado_distance = np.linalg.norm(ado_pos - ego_pos, axis=0)
        # if ego_ado_distance < RANGE_OVERTAKE:
        #     font = pygame.font.Font(pygame.font.get_default_font(), 42)
        #     text = font.render(self.ego_decision, True, (255, 255, 255), (0, 0, 0))
        #     text_rect = text.get_rect()
        #     text_rect.center = (WINDOW_W - 120, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        #     self.surf.blit(text, text_rect)

        #     font = pygame.font.Font(pygame.font.get_default_font(), 42)
        #     text = font.render(self.ai_acceptance, True, (255, 255, 255), (0, 0, 0))
        #     text_rect = text.get_rect()
        #     text_rect.center = (WINDOW_W - 500, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        #     self.surf.blit(text, text_rect)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            return self.create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        elif mode == "state_pixels":
            return self.create_image_array(self.surf, (STATE_W, STATE_H))
        else:
            return self.isopen

    def render_road(self, zoom, translation, angle):
        bounds = PLAYFIELD
        field = [
            (bounds, bounds),
            (bounds, -bounds),
            (-bounds, -bounds),
            (-bounds, bounds),
        ]

        # draw background
        self.draw_colored_polygon(field, self.bg_color, zoom, translation, angle, clip=False)

        # draw grass patches
        grass = []
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                grass.append(
                    [
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM),
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM),
                    ]
                )
        for poly in grass:
            self.draw_colored_polygon(poly, self.grass_color, zoom, translation, angle)

        # draw road
        for poly, color in self.road_poly:
            # converting to pixel coordinates
            poly = [(p[0], p[1]) for p in poly]
            color = [int(c) for c in color]
            self.draw_colored_polygon(poly, color, zoom, translation, angle)

    def render_indicators(self, car, W, H):
        s = W / 40.0
        h = H / 40.0
        color = (0, 0, 0)
        polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
        pygame.draw.polygon(self.surf, color=color, points=polygon)

        def vertical_ind(place, val):
            return [
                (place * s, H - (h + h * val)),
                ((place + 1) * s, H - (h + h * val)),
                ((place + 1) * s, H - h),
                ((place + 0) * s, H - h),
            ]

        def horiz_ind(place, val):
            return [
                ((place + 0) * s, H - 4 * h),
                ((place + val) * s, H - 4 * h),
                ((place + val) * s, H - 2 * h),
                ((place + 0) * s, H - 2 * h),
            ]

        assert car is not None
        true_speed = np.sqrt(np.square(car.hull.linearVelocity[0]) + np.square(car.hull.linearVelocity[1]))

        # simple wrapper to render if the indicator value is above a threshold
        def render_if_min(value, points, color):
            if abs(value) > 1e-4:
                pygame.draw.polygon(self.surf, points=points, color=color)

        render_if_min(true_speed, vertical_ind(5, 0.02 * true_speed), (255, 255, 255))
        # ABS sensors
        render_if_min(
            car.wheels[0].omega,
            vertical_ind(7, 0.01 * car.wheels[0].omega),
            (0, 0, 255),
        )
        render_if_min(
            car.wheels[1].omega,
            vertical_ind(8, 0.01 * car.wheels[1].omega),
            (0, 0, 255),
        )
        render_if_min(
            car.wheels[2].omega,
            vertical_ind(9, 0.01 * car.wheels[2].omega),
            (51, 0, 255),
        )
        render_if_min(
            car.wheels[3].omega,
            vertical_ind(10, 0.01 * car.wheels[3].omega),
            (51, 0, 255),
        )

        render_if_min(
            car.wheels[0].joint.angle,
            horiz_ind(20, -10.0 * car.wheels[0].joint.angle),
            (0, 255, 0),
        )
        render_if_min(
            car.hull.angularVelocity,
            horiz_ind(30, -0.8 * car.hull.angularVelocity),
            (255, 0, 0),
        )

    def draw_points(self, points, zoom, translation, angle, radius=5, color=(255, 0, 0)):
        points = [pygame.math.Vector2(*c).rotate_rad(angle) for c in points]
        points = [(c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in points]

        for point in points:
            if (0 <= point[0] < WINDOW_W) and (0 <= point[1] < WINDOW_H):
                gfxdraw.filled_circle(self.surf, int(point[0]), int(point[1]), radius, color)

    def draw_colored_polygon(self, poly, color, zoom, translation, angle, clip=True):
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [(c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly]
        # This checks if the polygon is out of bounds of the screen, and we skip drawing if so.
        # Instead of calculating exactly if the polygon and screen overlap,
        # we simply check if the polygon is in a larger bounding box whose dimension
        # is greater than the screen by MAX_SHAPE_DIM, which is the maximum
        # diagonal length of an environment object
        if not clip or any(
            (-MAX_SHAPE_DIM <= coord[0] <= WINDOW_W + MAX_SHAPE_DIM)
            and (-MAX_SHAPE_DIM <= coord[1] <= WINDOW_H + MAX_SHAPE_DIM)
            for coord in poly
        ):
            gfxdraw.aapolygon(self.surf, poly, color)
            gfxdraw.filled_polygon(self.surf, poly, color)

    def create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2))
