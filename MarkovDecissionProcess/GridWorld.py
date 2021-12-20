import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class GridWorld:
    def __init__(self, filename, reward=None):
        if reward is None:
            reward = {0: -0.04, 1: 1.0, 2: -1.0, 3: np.NaN}
        file = open(filename)
        self.map = np.array(
            [list(map(float, s.strip().split(","))) for s in file.readlines()]
        )
        self.num_rows = self.map.shape[0]
        self.num_cols = self.map.shape[1]
        self.num_states = self.num_rows * self.num_cols
        self.num_actions = 4
        self.reward = reward
        self.reward_function = self.get_reward_function()
        self.transition_model = self.get_transition_model()

    def get_state_from_pos(self, pos):
        return pos[0] * self.num_cols + pos[1]

    def get_pos_from_state(self, state):
        return state // self.num_cols, state % self.num_cols

    def get_reward_function(self):
        reward_table = np.zeros(self.num_states)
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                s = self.get_state_from_pos((r, c))
                reward_table[s] = self.reward[self.map[r, c]]
        return reward_table

    def get_transition_model(self, random_rate=0.2):
        transition_model = np.zeros((self.num_states, self.num_actions, self.num_states))
        for r in range(self.num_rows):
            for c in range(self.num_cols):
                s = self.get_state_from_pos((r, c))
                neighbor_s = np.zeros(self.num_actions)
                if self.map[r, c] == 0:
                    for a in range(self.num_actions):
                        new_r, new_c = r, c
                        if a == 0:
                            new_r = max(r - 1, 0)
                        elif a == 1:
                            new_c = min(c + 1, self.num_cols - 1)
                        elif a == 2:
                            new_r = min(r + 1, self.num_rows - 1)
                        elif a == 3:
                            new_c = max(c - 1, 0)
                        if self.map[new_r, new_c] == 3:
                            new_r, new_c = r, c
                        s_prime = self.get_state_from_pos((new_r, new_c))
                        neighbor_s[a] = s_prime
                else:
                    neighbor_s = np.ones(self.num_actions) * s
                for a in range(self.num_actions):
                    transition_model[s, a, int(neighbor_s[a])] += 1 - random_rate
                    transition_model[s, a, int(neighbor_s[(a + 1) % self.num_actions])] += random_rate / 2.0
                    transition_model[s, a, int(neighbor_s[(a - 1) % self.num_actions])] += random_rate / 2.0
        return transition_model

    def generate_random_policy(self):
        return np.random.randint(self.num_actions, size=self.num_states)

    def execute_policy(self, policy, start_pos=(2, 0)):
        s = self.get_state_from_pos(start_pos)
        r = self.reward_function[s]
        total_reward = r
        state_history = [s]
        while r != 1 and r != -1:
            temp = self.transition_model[s, policy[s]]
            # print(f'p = {temp}')
            s = np.random.choice(self.num_states, p=temp)
            state_history.append(s)
            r = self.reward_function[s]
            total_reward += r
            # print(f's = {s} r = {r}')
        return total_reward

    def random_start_policy(self, policy, n=10000):
        self.plot_policy(policy)
        policy_image = plt.imread('policy.png')

        scores = np.zeros(n)
        for i in range(n):
            print(f'i = {i}')
            scores[i] = self.execute_policy(policy=policy)

        print(f'max = {np.max(scores)}')
        print(f'min = {np.min(scores)}')
        print(f'mean = {np.mean(scores)}')
        print(f'std = {np.std(scores)}')

        bins = 100
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
        new_ax = fig.add_axes([0.15, 0.5, 0.4, 0.3], anchor='NW', zorder=1)
        new_ax.imshow(policy_image)
        new_ax.axis('off')
        ax.set_xlabel('Total rewards in a single game')
        ax.set_ylabel('Frequency')
        ax.hist(scores, bins=bins, color='#1f77b4', edgecolor='black')
        # plt.tight_layout()
        plt.show()

    def plot_reward(self):
        unit = 100
        fig, ax = plt.subplots(1, 1, figsize=(2 * self.num_cols, 2 * self.num_rows), dpi=300)
        ax.axis('off')
        # ax.set_title('State and Reward')
        for i in range(self.num_cols + 1):
            if i == 0 or i == self.num_cols:
                ax.plot([i * unit, i * unit], [0, self.num_rows * unit],
                        color='black')
            else:
                ax.plot([i * unit, i * unit], [0, self.num_rows * unit],
                        alpha=0.7, color='grey', linestyle='dashed')
        for i in range(self.num_rows + 1):
            if i == 0 or i == self.num_rows:
                ax.plot([0, self.num_cols * unit], [i * unit, i * unit],
                        color='black')
            else:
                ax.plot([0, self.num_cols * unit], [i * unit, i * unit],
                        alpha=0.7, color='grey', linestyle='dashed')

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                y = (self.num_rows - 1 - i) * unit
                x = j * unit
                s = self.get_state_from_pos((i, j))
                if self.map[i, j] == 3:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='black',
                                             alpha=0.6)
                    ax.add_patch(rect)
                elif self.map[i, j] == 2:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='red',
                                             alpha=0.6)
                    ax.add_patch(rect)
                elif self.map[i, j] == 1:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='green',
                                             alpha=0.6)
                    ax.add_patch(rect)

                ax.text(x + 0.5 * unit, y + 0.5 * unit, f's = {s}\nr = {self.reward_function[s]}',
                        horizontalalignment='center', verticalalignment='center',
                        fontsize=17)

        plt.tight_layout()
        plt.show()

    def plot_transition_model(self):
        unit = 100
        fig, ax = plt.subplots(1, 1, figsize=(2 * self.num_cols, 2 * self.num_rows), dpi=300)
        ax.axis('off')
        ax.set_title('Transitional model')
        for i in range(self.num_cols + 1):
            if i == 0 or i == self.num_cols:
                ax.plot([i * unit, i * unit], [0, self.num_rows * unit],
                        color='black')
            else:
                ax.plot([i * unit, i * unit], [0, self.num_rows * unit],
                        alpha=0.7, color='grey', linestyle='dashed')
        for i in range(self.num_rows + 1):
            if i == 0 or i == self.num_rows:
                ax.plot([0, self.num_cols * unit], [i * unit, i * unit],
                        color='black')
            else:
                ax.plot([0, self.num_cols * unit], [i * unit, i * unit],
                        alpha=0.7, color='grey', linestyle='dashed')

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                y = (self.num_rows - 1 - i) * unit
                x = j * unit
                s = self.get_state_from_pos((i, j))
                if self.map[i, j] == 3:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='black',
                                             alpha=0.6)
                    ax.add_patch(rect)
                elif self.map[i, j] == 2:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='red',
                                             alpha=0.6)
                    ax.add_patch(rect)
                elif self.map[i, j] == 1:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='green',
                                             alpha=0.6)
                    ax.add_patch(rect)
                if self.map[i, j] != 0:
                    ax.text(x + 0.5 * unit, y + 0.5 * unit, f's = {s}',
                            horizontalalignment='center', verticalalignment='center')
                else:
                    ax.text(x + 0.5 * unit, y + 0.5 * unit, f's = {s}',
                            horizontalalignment='center', verticalalignment='center')

                    string_transition = []
                    symbol = ['^', '>', 'v', '<']
                    for a in range(self.num_actions):
                        string_action = []
                        temp = self.transition_model[s, a]
                        for s_prime in range(len(temp)):
                            if temp[s_prime] > 0:
                                string_action.append(f's\' = {s_prime}, p = {temp[s_prime]}')
                        string_transition.append('\n'.join(string_action))

                    ax.text(x + 0.5 * unit, y + 0.97 * unit, f'{string_transition[0]}',
                            horizontalalignment='center', verticalalignment='top', fontsize=7)
                    ax.text(x + 0.97 * unit, y + 0.5 * unit, f'{string_transition[1]}',
                            horizontalalignment='right', verticalalignment='center', fontsize=7, rotation=90)
                    ax.text(x + 0.5 * unit, y + 0.03 * unit, f'{string_transition[2]}',
                            horizontalalignment='center', verticalalignment='bottom', fontsize=7)
                    ax.text(x + 0.03 * unit, y + 0.5 * unit, f'{string_transition[3]}',
                            horizontalalignment='left', verticalalignment='center', fontsize=7, rotation=90)

                    ax.plot([x + 0.5 * unit], [y + 0.85 * unit], marker=symbol[0], alpha=0.3,
                            linestyle='none', markersize=25, color='#1f77b4')
                    ax.plot([x + 0.85 * unit], [y + 0.5 * unit], marker=symbol[1], alpha=0.3,
                            linestyle='none', markersize=25, color='#1f77b4')
                    ax.plot([x + 0.5 * unit], [y + 0.15 * unit], marker=symbol[2], alpha=0.3,
                            linestyle='none', markersize=25, color='#1f77b4')
                    ax.plot([x + 0.15 * unit], [y + 0.5 * unit], marker=symbol[3], alpha=0.3,
                            linestyle='none', markersize=25, color='#1f77b4')

        plt.tight_layout()
        plt.show()

    def plot_policy(self, policy):
        unit = 100
        fig, ax = plt.subplots(1, 1, figsize=(2 * self.num_cols, 2 * self.num_rows), dpi=300)
        # ax.set_title('Policy 1')
        ax.axis('off')
        for i in range(self.num_cols + 1):
            if i == 0 or i == self.num_cols:
                ax.plot([i * unit, i * unit], [0, self.num_rows * unit],
                        color='black')
            else:
                ax.plot([i * unit, i * unit], [0, self.num_rows * unit],
                        alpha=0.7, color='grey', linestyle='dashed')
        for i in range(self.num_rows + 1):
            if i == 0 or i == self.num_rows:
                ax.plot([0, self.num_cols * unit], [i * unit, i * unit],
                        color='black')
            else:
                ax.plot([0, self.num_cols * unit], [i * unit, i * unit],
                        alpha=0.7, color='grey', linestyle='dashed')

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                y = (self.num_rows - 1 - i) * unit
                x = j * unit
                if self.map[i, j] == 3:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='black',
                                             alpha=0.6)
                    ax.add_patch(rect)
                elif self.map[i, j] == 2:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='red',
                                             alpha=0.6)
                    ax.add_patch(rect)
                elif self.map[i, j] == 1:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='green',
                                             alpha=0.6)
                    ax.add_patch(rect)
                s = self.get_state_from_pos((i, j))
                # ax.text(x + 0.5 * unit, y + 0.15 * unit, f's = {s}',
                #         horizontalalignment='center', verticalalignment='bottom')
                if self.map[i, j] == 0:
                    a = policy[s]
                    symbol = ['^', '>', 'v', '<']
                    ax.plot([x + 0.5 * unit], [y + 0.5 * unit], marker=symbol[a],
                            linestyle='none', markersize=30, color='#1f77b4')

        plt.tight_layout()
        plt.savefig('policy.png')

    def visualize_value_policy(self, policy, values):
        unit = 100
        fig, ax = plt.subplots(1, 1, figsize=(2 * self.num_cols, 2 * self.num_rows), dpi=300)
        ax.axis('off')
        # ax.set_title('Values')
        for i in range(self.num_cols + 1):
            if i == 0 or i == self.num_cols:
                ax.plot([i * unit, i * unit], [0, self.num_rows * unit],
                        color='black')
            else:
                ax.plot([i * unit, i * unit], [0, self.num_rows * unit],
                        alpha=0.7, color='grey', linestyle='dashed')
        for i in range(self.num_rows + 1):
            if i == 0 or i == self.num_rows:
                ax.plot([0, self.num_cols * unit], [i * unit, i * unit],
                        color='black')
            else:
                ax.plot([0, self.num_cols * unit], [i * unit, i * unit],
                        alpha=0.7, color='grey', linestyle='dashed')

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                y = (self.num_rows - 1 - i) * unit
                x = j * unit
                s = self.get_state_from_pos((i, j))
                if self.map[i, j] == 3:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='black',
                                             alpha=0.6)
                    ax.add_patch(rect)
                elif self.map[i, j] == 2:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='red',
                                             alpha=0.6)
                    ax.add_patch(rect)
                elif self.map[i, j] == 1:
                    rect = patches.Rectangle((x, y), unit, unit, edgecolor='none', facecolor='green',
                                             alpha=0.6)
                    ax.add_patch(rect)
                if self.map[i, j] != 3:
                    ax.text(x + 0.5 * unit, y + 0.5 * unit, f's = {s}\n v = {values[s]:.4f}',
                            horizontalalignment='center', verticalalignment='center',
                            fontsize=16)
                if policy is not None:
                    if self.map[i, j] == 0:
                        a = policy[s]
                        symbol = ['^', '>', 'v', '<']
                        ax.plot([x + 0.5 * unit], [y + 0.5 * unit], marker=symbol[a], alpha=0.4,
                                linestyle='none', markersize=45, color='#1f77b4')

        plt.tight_layout()
        plt.show()




