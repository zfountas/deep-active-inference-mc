import time
import numpy as np
import torch
from src.util import np_precision

class Game:
    def __init__(self, games_no):
        self.games_no = games_no  # Ensure games_no is initialized
        current_time = time.time()
        dataset = np.load('dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', allow_pickle=True, encoding='latin1')
        self.imgs = torch.from_numpy(dataset['imgs'].reshape(-1, 64, 64, 1))
        latents_values = dataset['latents_values']  # Normalized version of classes..
        latents_classes = dataset['latents_classes']
        metadata = dataset['metadata'][()]
        self.s_sizes = torch.from_numpy(metadata['latents_sizes'])  # [1 3 6 40 32 32]
        self.s_dim = max(self.s_sizes.size(0) + 1, 7)  # Ensure self.s_dim is correctly set

        # Initialize current_s with the correct dimensions
        self.current_s = torch.zeros((self.games_no, self.s_dim), dtype=torch.float32)
        self.last_r = torch.zeros(self.games_no, dtype=torch.float32)
        self.new_image_all()
        print('Dataset loaded. Time:', time.time() - current_time, 'datapoints:', len(self.imgs), self.s_dim)

        # Add this line to initialize s_bases
        self.s_bases = torch.tensor([1, 3, 6, 40, 32, 32])  # Adjust the values as needed

    def sample_s(self):  # Reward is zero after this!
        s = torch.zeros(self.s_dim, dtype=torch.float32)
        for s_i, s_size in enumerate(self.s_sizes):
            s[s_i] = torch.randint(s_size, (1,)).item()
        return s

    def sample_s_all(self):  # Reward is zero after this!
        s = torch.zeros((self.games_no, self.s_dim), dtype=torch.float32)  # Ensure s is a tensor
        for s_i in range(min(self.s_sizes.size(0), self.s_dim)):
            s[:, s_i] = torch.randint(0, self.s_sizes[s_i], (self.games_no,))
        return s

    def s_to_index(self, s):
        # Convert s to the same dtype as self.s_bases
        s = s.to(dtype=self.s_bases.dtype)
        return torch.dot(s, self.s_bases).long()

    def s_to_o(self, index):
        image_to_return = self.imgs[self.s_to_index(self.current_s[index, :-1])].float()

        # Adding the reward encoded to the image..
        if 0.0 <= self.last_r[index] <= 1.0:
            image_to_return[0:3, 0:32] = self.last_r[index]
        elif -1.0 <= self.last_r[index] < 0.0:
            image_to_return[0:3, 32:64] = -self.last_r[index]
        else:
            raise ValueError(f'Error: Reward: {self.last_r[index]}')
        return image_to_return

    def reward_to_rgb(self, reward):
        return torch.tensor([min(1.0, -reward+1), min(1.0, reward+1), 1.0 - abs(reward)], dtype=torch.float32)

    def current_frame(self, index):
        return self.s_to_o(index)

    def current_frame_all(self):
        o = torch.zeros((self.games_no, 64, 64, 1), dtype=torch.float32)
        for i in range(self.games_no):
            o[i] = self.s_to_o(i)
        return o

    def randomize_environment(self, index):
        self.current_s[index] = self.sample_s()
        self.current_s[index, 6] = -10 + torch.rand(1).item() * 20
        self.last_r[index] = -1.0 + torch.rand(1).item() * 2.0

    def randomize_environment_all(self):
        self.current_s = self.sample_s_all()
        self.current_s[:, 6] = -10 + torch.rand(self.games_no) * 20
        self.last_r = -1.0 + torch.rand(self.games_no) * 2.0

    def new_image(self, index):
        reward = self.current_s[index, 6]  # pass reward to the new latent..!
        self.current_s[index] = self.sample_s()
        self.current_s[index, 6] = reward

    def new_image_all(self):
        reward = self.current_s[:, 6] if self.current_s.shape[1] > 6 else torch.zeros(self.games_no)
        self.current_s = self.sample_s_all()
        if self.current_s.shape[1] > 6:
            self.current_s[:, 6] = reward
        print(f"Shape of self.current_s: {self.current_s.shape}")  # Debugging line

    def get_reward(self, index):
        return self.current_s[index, 6]

    # NOTE: Randomness takes values from zero to one.
    def find_move(self, index, randomness):
        right = 0.5 * (1.0 - randomness / 2.0)
        wrong = 0.5 * randomness / 2.0
        if self.current_s[index, 1] < 0.5:  # Square
            Ppi = torch.tensor([right, wrong, wrong, right], dtype=torch.float32)
        else:  # Ellipse or heart
            Ppi = torch.tensor([right, wrong, right, wrong], dtype=torch.float32)
        return Ppi

    def find_move_all(self, randomness):
        return torch.stack([self.find_move(i, randomness) for i in range(self.games_no)])

    # NOTE: Randomness takes values from zero to one.
    def auto_play(self, index, randomness=0.4):
        Ppi = self.find_move(index, randomness)
        pi = torch.multinomial(Ppi, 1).item()
        self.pi_to_action(pi, index)
        return pi, Ppi

    def tick(self, index):
        self.last_r[index] *= 0.95

    def tick_all(self):
        self.last_r *= 0.95

    def up(self, index):
        self.tick(index)
        self.current_s[index, 5] += 1.0
        if self.current_s[index, 5] >= 32:
            if self.current_s[index, 1] < 0.5:  # Square
                if self.current_s[index, 4] > 15:
                    self.last_r[index] = float(15.0 - self.current_s[index, 4]) / 16.0
                else:
                    self.last_r[index] = float(16.0 - self.current_s[index, 4]) / 16.0
                self.current_s[index, 6] += self.last_r[index]
            else:  # Ellipse or heart
                if self.current_s[index, 4] > 15:
                    self.last_r[index] = float(self.current_s[index, 4] - 15.0) / 16.0
                else:
                    self.last_r[index] = float(self.current_s[index, 4] - 16.0) / 16.0
                self.current_s[index, 6] += self.last_r[index]
            self.new_image(index)
            return True
        return False

    def down(self, index):
        self.tick(index)
        if self.current_s[index, 5] > 0:
            self.current_s[index, 5] -= 1.0

    def left(self, index):
        self.tick(index)
        if self.current_s[index, 4] < 31:
            self.current_s[index, 4] += 1.0

    def right(self, index):
        self.tick(index)
        if self.current_s[index, 4] > 0:
            self.current_s[index, 4] -= 1.0

    def pi_to_action(self, pi, index, repeats=1):
        round_changed = False
        for i in range(repeats):
            if pi == 0:
                round_changed = self.up(index)
            elif pi == 1:
                self.down(index)
            elif pi == 2:
                self.left(index)
            elif pi == 3:
                self.right(index)
            else:
                raise ValueError('Invalid action')
            if round_changed:
                return True
        return False
