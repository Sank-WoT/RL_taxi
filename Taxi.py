import gym
import numpy as np
import time

env = gym.make('Taxi-v2')

alpha = 0.1  # Скорость обучения
gamma = 0.9  # Коэффициент дисконтирования
epsilon = 0.1  # Вероятность случайного выбора действия

num_episodes = 10000


class Agent():
    def __init__(self, env, alpha, gamma, epsilon, num_episodes):
        self.env = env
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes

    def fit(self):
        for episode in range(self.num_episodes):
            state = self.env.reset()
            done = False

            while not done:
                if np.random.rand() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state])

                next_state, reward, done, info = self.env.step(action)

                self.q_table[state, action] = (1 - self.alpha) * self.q_table[state, action] + self.alpha * (
                        reward + self.gamma * np.max(self.q_table[next_state]))

                state = next_state

        return self.q_table


agent = Agent(env, alpha, gamma, epsilon, num_episodes)
q_table = agent.fit()

# Оценка агента
num_tests = 10
total_reward = 0

for _ in range(num_tests):
    state = env.reset()
    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)
        total_reward += reward

        time.sleep(0.5)
        env.render()

print(f"Средняя награда за {num_tests} эпизодов: {total_reward / num_tests}")
