import gym
import numpy as np
from gym.utils import seeding
from gym import spaces
from pong_player import PongPlayer


class PongEnv(gym.Env):

    def __init__(self):

        self.ball_max_vel = 0.1
        self.min_x_mag = 0.025
        self.paddle_max_vel = 0.1
        self.paddle_height = 1. / 8
        self.paddle_width = 1. / 60
        self.max_score = 20

        # state is tuple: (ballx, bally, ballvx, ballvy, paddle1y, paddle1vy, paddle2y, paddle2vy, score1, score2)
        self.state_low = np.array([
            0.0, 0.0, -self.ball_max_vel, -self.ball_max_vel, 0.0, -self.paddle_max_vel, 0.0, -self.paddle_max_vel, 0, 0
        ])

        self.state_high = np.array([
            1.0, 1.0, self.ball_max_vel, self.ball_max_vel, 1.0, self.paddle_max_vel, 1.0, self.paddle_max_vel, self.max_score, self.max_score
        ])

        self.viewer = None
        self.player1_state = None
        self.player2_state = None

        self.player1_action_space = spaces.Discrete(3)
        self.player2_action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.state_low, self.state_high)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, player1_action, player2_action):
        print ('Player 1', player1_action)
        print ('Player 2', player2_action)
        assert self.player1_action_space.contains(player1_action), "Invalid action player 1."
        assert self.player2_action_space.contains(player2_action), "Invalid action player 2."

        ball_pos_x, ball_pos_y, ball_vel_x, ball_vel_y, paddle1_pos_y, paddle1_vel_y, score1 = \
            self.player1_state
        ball_pos_x, ball_pos_y, ball_vel_x, ball_vel_y, paddle2_pos_y, paddle2_vel_y, score2 = \
            self.player2_state

        player1_reward = 0.0
        player2_reward = 0.0

        if ball_pos_x == 1.0:
            # collision with right (player2 side)
            if ball_pos_y == 0.0 or ball_pos_y == 1.0:
                print('Stuck in corner. Player 1 scores point.')
                player2_reward = -10.0
                player1_reward = 10.0
                score1 += 1
                ball_pos_x = 0.5
                ball_pos_y = 0.5
                ball_vel_x = (self.np_random.rand() - 0.5) * self.ball_max_vel
                ball_vel_y = (self.np_random.rand() - 0.5) * self.ball_max_vel
                paddle1_pos_y = 0.5
                paddle1_vel_y = 0.0
                paddle2_pos_y = 0.5
                paddle2_vel_y = 0.0
                self.player1_state = [
                    ball_pos_x,
                    ball_pos_y,
                    ball_vel_x,
                    ball_vel_y,
                    paddle1_pos_y,
                    paddle1_vel_y,
                    score1,
                ]
                self.player2_state = [
                    ball_pos_x,
                    ball_pos_y,
                    ball_vel_x,
                    ball_vel_y,
                    paddle2_pos_y,
                    paddle2_vel_y,
                    score2,
                ]
                done = bool(score1 >= self.max_score or score2 >= self.max_score)
                return np.array(self.player1_state), np.array(self.player2_state), player1_reward, player2_reward, done, {}

            if paddle2_pos_y - self.paddle_height < ball_pos_y and paddle2_pos_y + self.paddle_height > ball_pos_y:
                # collision with player paddle
                print('Player 2 hit ball.')
                player2_reward = 1.0                
                ball_vel_x = -ball_vel_x
            
            else:
                print('Player 1 scores point.')
                player2_reward = -10.0
                player1_reward = 10.0
                score1 += 1
                ball_pos_x = 0.5
                ball_pos_y = 0.5
                ball_vel_x = (self.np_random.rand() - 0.5) * self.ball_max_vel
                ball_vel_y = (self.np_random.rand() - 0.5) * self.ball_max_vel
                paddle1_pos_y = 0.5
                paddle1_vel_y = 0.0
                paddle2_pos_y = 0.5
                paddle2_vel_y = 0.0
                self.player1_state = [
                    ball_pos_x,
                    ball_pos_y,
                    ball_vel_x,
                    ball_vel_y,
                    paddle1_pos_y,
                    paddle1_vel_y,
                    score1,
                ]
                self.player2_state = [
                    ball_pos_x,
                    ball_pos_y,
                    ball_vel_x,
                    ball_vel_y,
                    paddle2_pos_y,
                    paddle2_vel_y,
                    score2,
                ]
                done = bool(score1 >= self.max_score or score2 >= self.max_score)
                return np.array(self.player1_state), np.array(self.player2_state), player1_reward, player2_reward, done, {}
 
            # collision with right
            ball_vel_x = -ball_vel_x

        elif ball_pos_x == 0.0:
            if ball_pos_y == 0.0 or ball_pos_y == 1.0:
                print('Stuck in corner. Player 2 scores point.')
                player1_reward = -10.0
                player2_reward = 10.0
                score2 += 1
                ball_pos_x = 0.5
                ball_pos_y = 0.5
                ball_vel_x = (self.np_random.rand() - 0.5) * self.ball_max_vel
                ball_vel_y = (self.np_random.rand() - 0.5) * self.ball_max_vel
                paddle1_pos_y = 0.5
                paddle1_vel_y = 0.0
                paddle2_pos_y = 0.5
                paddle2_vel_y = 0.0
                self.player1_state = [
                    ball_pos_x,
                    ball_pos_y,
                    ball_vel_x,
                    ball_vel_y,
                    paddle1_pos_y,
                    paddle1_vel_y,
                    score1,
                ]
                self.player2_state = [
                    ball_pos_x,
                    ball_pos_y,
                    ball_vel_x,
                    ball_vel_y,
                    paddle2_pos_y,
                    paddle2_vel_y,
                    score2,
                ]
                done = bool(score1 >= self.max_score or score2 >= self.max_score)
                return np.array(self.player1_state), np.array(self.player2_state), player1_reward, player2_reward, done, {}

            # collision with left (player1 side)
            if paddle1_pos_y - self.paddle_height < ball_pos_y and paddle1_pos_y + self.paddle_height > ball_pos_y:
                # collision with player paddle
                print('Player 1 hit ball.')
                reward = 1.0                
                ball_vel_x = -ball_vel_x
            else:
                print('Player 2 scores point.')
                player1_reward = -10.0
                player2_reward = 10.0
                score2 += 1
                ball_pos_x = 0.5
                ball_pos_y = 0.5
                ball_vel_x = (self.np_random.rand() - 0.5) * self.ball_max_vel
                ball_vel_y = (self.np_random.rand() - 0.5) * self.ball_max_vel
                paddle1_pos_y = 0.5
                paddle1_vel_y = 0.0
                paddle2_pos_y = 0.5
                paddle2_vel_y = 0.0
                self.player1_state = [
                    ball_pos_x,
                    ball_pos_y,
                    ball_vel_x,
                    ball_vel_y,
                    paddle1_pos_y,
                    paddle1_vel_y,
                    score1,
                ]
                self.player2_state = [
                    ball_pos_x,
                    ball_pos_y,
                    ball_vel_x,
                    ball_vel_y,
                    paddle2_pos_y,
                    paddle2_vel_y,
                    score2,
                ]
                done = bool(score1 >= self.max_score or score2 >= self.max_score)
                return np.array(self.player1_state), np.array(self.player2_state), player1_reward, player2_reward, done, {}

        elif (ball_pos_y == 1.0 or ball_pos_y == 0.0):
            # collision with top or bottom
            ball_vel_y = -ball_vel_y


        done = bool(score1 >= self.max_score or score2 >= self.max_score)
        
        ball_pos_x += ball_vel_x
        ball_pos_y += ball_vel_y

        paddle1_vel_y += 0.05 * (player1_action - 1)
        paddle1_vel_y = np.clip(paddle1_vel_y, -self.paddle_max_vel, self.paddle_max_vel)
        paddle1_pos_y += paddle1_vel_y

        paddle2_vel_y += 0.05 * (player2_action - 1)
        paddle2_vel_y = np.clip(paddle2_vel_y, -self.paddle_max_vel, self.paddle_max_vel)
        paddle2_pos_y += paddle2_vel_y


        ball_pos_x = np.clip(ball_pos_x, 0.0, 1.0)
        ball_pos_y = np.clip(ball_pos_y, 0.0, 1.0)
        paddle1_pos_y = np.clip(paddle1_pos_y, 0.0, 1.0)
        paddle2_pos_y = np.clip(paddle2_pos_y, 0.0, 1.0)
        ball_vel_x = np.clip(ball_vel_x, -self.ball_max_vel, self.ball_max_vel)
        ball_vel_y = np.clip(ball_vel_y, -self.ball_max_vel, self.ball_max_vel)

        if np.abs(ball_vel_x) < self.min_x_mag:
            ball_vel_x = np.sign(ball_vel_x) * self.min_x_mag

        self.player1_state = [
            ball_pos_x,
            ball_pos_y,
            ball_vel_x,
            ball_vel_y,
            paddle1_pos_y,
            paddle1_vel_y,
            score1,
        ]
        self.player2_state = [
            ball_pos_x,
            ball_pos_y,
            ball_vel_x,
            ball_vel_y,
            paddle2_pos_y,
            paddle2_vel_y,
            score2,
        ]
        return np.array(self.player1_state), np.array(self.player2_state), player1_reward, player2_reward, done, {}

    def reset(self):
        ball_pos_x = 0.5
        ball_pos_y = 0.5
        ball_vel_x = (self.np_random.rand() - 0.5) * self.ball_max_vel
        ball_vel_y = (self.np_random.rand() - 0.5) * self.ball_max_vel
        paddle1_pos_y = 0.5
        paddle1_vel_y = 0.0
        paddle2_pos_y = 0.5
        paddle2_vel_y = 0.0
        score1 = 0
        score2 = 0

        self.player1_state = [
            ball_pos_x,
            ball_pos_y,
            ball_vel_x,
            ball_vel_y,
            paddle1_pos_y,
            paddle1_vel_y,
            score1,
        ]
        self.player2_state = [
            ball_pos_x,
            ball_pos_y,
            ball_vel_x,
            ball_vel_y,
            paddle2_pos_y,
            paddle2_vel_y,
            score2,
        ]
        return self.player1_state, self.player2_state
        
    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400
        world_width = 1.0
        world_height = 1.0
        xscale = screen_width / world_width
        yscale = screen_height / world_height

        paddle_width = xscale * self.paddle_width
        paddle_height = yscale * self.paddle_height
        ball_radius = 10

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            l,r,t,b = -paddle_width/2, paddle_width/2, paddle_height, 0
            paddle1 = rendering.FilledPolygon([
                (l,b), (l,t), (r,t), (r,b)
            ])
            paddle1.set_color(0.5, 0.5, 0.5)

            self.paddle1_transform = rendering.Transform()
            paddle1.add_attr(self.paddle1_transform)

            self.viewer.add_geom(paddle1)

            l,r,t,b = -paddle_width/2, paddle_width/2, paddle_height, 0.0
            paddle2 = rendering.FilledPolygon([
                (l,b), (l,t), (r,t), (r,b)
            ])
            paddle2.set_color(0.5, 0.5, 0.5)

            self.paddle2_transform = rendering.Transform()
            paddle2.add_attr(self.paddle2_transform)

            self.viewer.add_geom(paddle2)

            ball = rendering.make_circle(ball_radius)
            ball.set_color(0.5, 0.5, 0.5)

            self.ball_transform = rendering.Transform()
            ball.add_attr(self.ball_transform)

            self.viewer.add_geom(ball)

        ball_pos_x, ball_pos_y, _, _, paddle1_pos_y, _, _ = self.player1_state
        ball_pos_x, ball_pos_y, _, _, paddle2_pos_y, _, _ = self.player2_state
        #self.paddle_transform.set_translation(50, 50)
        self.paddle1_transform.set_translation(paddle_width/2, paddle1_pos_y * yscale)
        self.paddle2_transform.set_translation(screen_width - paddle_width/2, paddle2_pos_y * yscale)
        self.ball_transform.set_translation(ball_pos_x * xscale, ball_pos_y * yscale)
        #self.ball_transform.set_translation(50, 50)

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == '__main__':
    env = PongEnv()
    env.reset()
    done = False
    p1_state, p2_state = env.reset()
    player1 = PongPlayer('./player1')
    player2 = PongPlayer('./player2')
    player1.load()
    player2.load()
    p1_action = player1.get_action(p1_state)
    p2_action = player2.get_action(p2_state)
    p1_total_reward = 0
    p2_total_reward = 0
    p1_reward = 0
    p2_reward = 0
    p1_memory = []
    p2_memory = []

    while not done:
        p1_current = np.array(p1_state[:])
        print('P1 current:', type(p1_current))
        p1_current = np.append(p1_current, p1_action)
        p1_current = np.append(p1_current, p1_reward)
        p1_memory.append(p1_current[:])

        p2_current = np.array(p2_state[:])
        print('P2 current:', type(p2_current))
        p2_current = np.append(p2_current, p2_action)
        p2_current = np.append(p2_current, p2_reward)
        p2_memory.append(p2_current[:])

        p1_state, p2_state, p1_reward, p2_reward, done, _ = env.step(p1_action, p2_action)
        env.render()
        p1_total_reward += p1_reward
        p2_total_reward += p2_reward
        p1_action = player1.get_action(p1_state)
        p2_action = player2.get_action(p2_state)
        player1.learn(np.array(p1_memory[:]), np.array(p1_state[:]))
        player2.learn(np.array(p2_memory)[:], np.array(p2_state[:]))
    
    print('Reward: ', p1_total_reward, ', ', p2_total_reward)
    player1.save()
    player2.save()

    player1.reset()
    player2.reset()
    
    env.close()