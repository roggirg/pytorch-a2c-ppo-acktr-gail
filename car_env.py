import random
import sys, math
import time

import numpy as np

import Box2D
import pygame
from car_dynamics import Car
from pygame.locals import VIDEORESIZE
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

# Rocket trajectory optimization is a classic topic in Optimal Control.
#
# According to Pontryagin's maximum principle it's optimal to fire engine full throttle or
# turn it off. That's the reason this environment is OK to have discreet actions (engine on or off).
#
# Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector.
# Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points.
# If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or
# comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main
# engine is -0.3 points each frame. Firing side engine is -0.03 points each frame. Solved is 200 points.
#
# Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land
# on its first attempt. Please see source code for details.
#
# To see heuristic landing, run:
#
# python gym/envs/box2d/lunar_lander.py
#
# To play yourself, run:
#
# python examples/agents/keyboard_agent.py LunarLander-v2
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
from gym.utils.play import display_arr

FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well
VIEWPORT_W = 1000
VIEWPORT_H = 800


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.agent.hull in [contact.fixtureA.body, contact.fixtureB.body] and \
                self.env.road in [contact.fixtureA.body, contact.fixtureB.body]:
            self.env.game_over = True
            self.env.sidewalk_crash = True

        if self.env.include_opponent:
            if self.env.agent.hull in [contact.fixtureA.body, contact.fixtureB.body] and \
                    self.env.othercar.hull in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.game_over = True
                self.env.car_crash = True

    def EndContact(self, contact):
        pass


class CarEnv(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    include_opponent = False
    discrete_ations = False

    def __init__(self):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World(gravity=(0, 0))
        self.road = None
        self.prev_reward = None
        self.agent = None
        self.othercar = None
        self.day_type = None
        self.road_polys = None
        self.friction_limit = None

        # useful range is -1 .. +1, but spikes can be higher
        if not self.include_opponent:
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32)

        if not self.discrete_ations:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (3,), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(5)

        self.max_episode_length = 300

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.road: return
        self.world.contactListener = None
        self.world.DestroyBody(self.road)
        self.road = None
        self.agent.destroy()
        self.agent = None
        if self.include_opponent:
            self.othercar.destroy()
        self.othercar = None

    def reset(self):
        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.car_crash = False
        self.sidewalk_crash = False
        self.prev_shaping = None

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        # Vertical Street
        p1 = (0.35*W, 0)
        p2 = (0.65*W, 0)
        # Horizontal Street
        p3 = (0, 0.20*H)
        p4 = (0, 0.55*H)
        self.road = self.world.CreateStaticBody(position=(0, 0))
        self.road.CreateEdgeFixture(vertices=[p1, (p1[0], p3[1])], density=0.1, friction=0.1)
        self.road.CreateEdgeFixture(vertices=[p2, (p2[0], p3[1])], density=0.1, friction=0.1)
        self.road.CreateEdgeFixture(vertices=[p3, (p1[0], p3[1])], density=0.1, friction=0.1)
        self.road.CreateEdgeFixture(vertices=[(p2[0], p3[1]), (W, p3[1])], density=0.1, friction=0.1)
        self.road.CreateEdgeFixture(vertices=[p4, (p1[0], p4[1])], density=0.1, friction=0.1)
        self.road.CreateEdgeFixture(vertices=[(p2[0], p4[1]), (W, p4[1])], density=0.1, friction=0.1)
        self.road.CreateEdgeFixture(vertices=[(p1[0], p4[1]), (p1[0], H)], density=0.1, friction=0.1)
        self.road.CreateEdgeFixture(vertices=[(p2[0], p4[1]), (p2[0], H)], density=0.1, friction=0.1)

        self.road_polys = []
        self.road_polys.append([p1, p2, (p2[0], H), (p1[0], H)])
        self.road_polys.append([p3, p4, (W, p4[1]), (W, p3[1])])
        self.road_polys.append([(0, 0), p1, (p1[0], p3[1]), p3])
        self.road_polys.append([(W, 0), p2, (p2[0], p3[1]), (W, p3[1])])
        self.road_polys.append([(0, H), p4, (p1[0], p4[1]), (p1[0], H)])
        self.road_polys.append([(W, H), (p2[0], H), (p2[0], p4[1]), (W, p4[1])])

        self.day_type = self.np_random.choice([0, 1, 2], 1, p=[0.3, 0.3, 0.4])
        if self.day_type == 0:  # normal
            self.friction_limit = 9500000
            self.road_color = (0.5, 0.5, 0.5)
            self.env_color = (0.1, 0.7, 0.1)
            self.slope_bias = (0.055/-20000, 0.65)
            self.decision_thresh = (50000, 65000)
        elif self.day_type == 1:  # rainy
            self.friction_limit = 2000000
            self.road_color = (0.3, 0.3, 0.3)
            self.env_color = (0.0, 0.7, 1.0)
            self.slope_bias = (0.05/-15000, 0.65)
            self.decision_thresh = (45000, 55000)
        else:  # icy
            self.friction_limit = 1100000
            self.road_color = (0.3, 0.3, 0.3)
            self.env_color = (1.0, 1.0, 1.0)
            self.slope_bias = (0.04 / -10000, 0.69)
            self.decision_thresh = (40000, 45000)

        self.agent = Car(self.world, 0.0, 0.55*W, 0.05*H, self.friction_limit)
        self.agent.hull.color = (1.0, 0.5, 0.0)

        if self.include_opponent:
            self.create_other_car()
        else:
            self.t = 0

        return self.step(np.array([0, 0, 0]))[0]

    def step(self, action):
        self.t += 1
        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        if self.discrete_ations:
            # discrete_actions: 0=nothing, 1=forward, 2=stop, 3=left, 4=right
            # if k == key.LEFT:  a[0] = -1.0
            # if k == key.RIGHT: a[0] = +1.0
            # if k == key.UP:    a[1] = +1.0
            # if k == key.DOWN:  a[2] = +0.8
            discrete_action = action
            action = np.array([0, 0, 0])
            if discrete_action == 1:
                action[1] = 1.0
            elif discrete_action == 2:
                action[2] = 0.8
            elif discrete_action == 3:
                action[0] = -1.0
            elif discrete_action == 4:
                action[0] = 1.0

        if action is not None:
            self.agent.steer(-action[0])
            self.agent.gas(action[1])
            self.agent.brake(action[2])
        self.agent.step(1.0/FPS)

        if self.include_opponent:
            othercar_y = self.othercar.hull.position.y / H

            if self.othercar_choice == 0:  # If stop
                if othercar_y <= self.stop_location:
                    self.othercar.gas(0.0)
                    self.othercar.brake(0.8)
            elif self.othercar_choice == 1:
                self.othercar.gas(1.0)
                self.othercar.brake(0.0)
            else:
                if self.go_location <= othercar_y <= self.stop_location:
                    self.othercar.gas(0.0)
                    self.othercar.brake(0.8)
                else:
                    self.othercar.gas(1.0)
                    self.othercar.brake(0.0)

            self.othercar.steer(0.0)
            self.othercar.step(1.0/FPS)

        self.world.Step(1.0/FPS, 6*FPS, 2*FPS)
        # self.state = self.render("state_pixels")

        if self.include_opponent:
            if othercar_y <= 0.05:  # Has gone out of screen
                # Destroy old car and create new one
                self.create_other_car()

        agent_x = self.agent.hull.position.x / W
        agent_y = self.agent.hull.position.y / H
        if not self.include_opponent:
            state = [
                agent_x,
                agent_y,
                self.agent.hull.linearVelocity.x,
                self.agent.hull.linearVelocity.y,
                self.day_type
            ]
        else:
            state = [
                agent_x,
                agent_y,
                self.agent.hull.linearVelocity.x,
                self.agent.hull.linearVelocity.y,
                self.othercar.hull.position.x / W,
                self.othercar.hull.position.y / H,
                self.othercar.hull.linearVelocity.x,
                self.othercar.hull.linearVelocity.y,
                self.day_type
            ]

        if agent_x <= 0.0:
            reward = 1.0
            done = True
        elif agent_x >= 1.0 or agent_y >= 1.0 or agent_y <= 0.0:
            reward = -1.0
            done = True
        else:
            # Yellow light idea. Need to go asap.
            if self.t <= 100:
                reward = -0.01
            else:
                reward = -0.05
            done = False

        if self.game_over:
            done = True
            if self.sidewalk_crash:
                reward = -0.5
            if self.car_crash:
                reward = -0.75

        if not done and self.t >= self.max_episode_length:
            done = True

        return np.array(state, dtype=np.float32), reward, done, {}

    def create_other_car(self):
        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE
        if self.othercar is not None:
            self.othercar.destroy()
        self.othercar = Car(self.world, math.pi, 0.45 * W, 1.0 * H, self.friction_limit)
        self.othercar.hull.color = (self.np_random.uniform(0, 1),
                                    self.np_random.uniform(0, 1),
                                    self.np_random.uniform(0, 1))

        applied_force = (0, self.np_random.uniform(-30000, -80000))
        # print("Applied Force:", applied_force[1])
        self.othercar.hull.ApplyForceToCenter(applied_force, True)
        if abs(applied_force[1]) <= self.decision_thresh[0]:
            self.othercar_choice = 0  # Stop at line
            self.stop_location = (self.slope_bias[0]*(applied_force[1]+30000)) + self.slope_bias[1]
        elif abs(applied_force[1]) > self.decision_thresh[1]:
            self.othercar_choice = 1  # Keep going
        else:
            self.othercar_choice = 2  # stop, and maybe keep going. 50/50
            self.stop_location = self.slope_bias[1]
            self.go_location = self.np_random.choice([0.0, 0.6], 1, p=[0.5, 0.5])

        self.t = 0

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)

        for i, p in enumerate(self.road_polys):
            if i < 2:
                color = self.road_color
            else:
                color = self.env_color
            self.viewer.draw_polygon(p, color=color)

        self.agent.draw(self.viewer,  draw_particles=False)
        if self.include_opponent:
            self.othercar.draw(self.viewer, draw_particles=False)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class CarEnvWithOpponent(CarEnv):
    include_opponent = True


class CarEnvDiscrete(CarEnv):
    discrete_actions = True


if __name__ == '__main__':
    # demo_heuristic_lander(CarEnv(), render=True, action_repeat=2)
    # manual_control(CarEnv(), action_repeat=2)
    from pyglet.window import key
    a = np.array( [0.0, 0.0, 0.0] )

    def key_press(k, mod):
        global restart
        if k==0xff0d: restart = True
        if k==key.LEFT:  a[0] = -1.0
        if k==key.RIGHT: a[0] = +1.0
        if k==key.UP:    a[1] = +1.0
        if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation


    def key_release(k, mod):
        if k==key.LEFT  and a[0]==-1.0: a[0] = 0
        if k==key.RIGHT and a[0]==+1.0: a[0] = 0
        if k==key.UP:    a[1] = 0
        if k==key.DOWN:  a[2] = 0


    env = CarEnv()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor
        env = Monitor(env, '/tmp/video-test', force=True)
    isopen = True
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                print("Total timesteps", env.t)
                break
    env.close()
