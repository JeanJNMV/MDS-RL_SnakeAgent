from __future__ import annotations

import copy
import random
from abc import ABC, abstractmethod
from collections import deque

import numpy as np
import torch
from torch import nn, optim

from rl_snake.env import DOWN, LEFT, RIGHT, UP, SnakeEnv

# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------

_TURN_RIGHT = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}
_TURN_LEFT = {UP: LEFT, RIGHT: UP, DOWN: RIGHT, LEFT: DOWN}
_DELTA = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}


def _is_dangerous(env: SnakeEnv, pos: tuple[int, int]) -> bool:
    r, c = pos
    if r < 0 or r >= env.height or c < 0 or c >= env.width:
        return True
    if pos in env.all_obstacle_positions:
        return True
    # tail will move away this step, so exclude it
    return pos in env.snake[:-1]


def _nearest_food_of_type(
    env: SnakeEnv, food_types: tuple[str, ...]
) -> tuple[int, int] | None:
    """Return the position of the nearest food matching any of the given types."""
    hr, hc = env.snake[0]
    best_pos = None
    best_dist = float("inf")
    for pos, ftype in env.foods.items():
        if ftype in food_types:
            d = abs(pos[0] - hr) + abs(pos[1] - hc)
            if d < best_dist:
                best_dist = d
                best_pos = pos
    return best_pos


def get_state(env: SnakeEnv) -> np.ndarray:
    """Return the 15-feature state vector.

    Features (all binary floats):
        [0]   danger straight ahead
        [1]   danger to the right (relative turn)
        [2]   danger to the left  (relative turn)
        [3]   moving up
        [4]   moving right
        [5]   moving down
        [6]   moving left
        [7]   nearest positive food (gold or silver) is above head
        [8]   nearest positive food is below head
        [9]   nearest positive food is left of head
        [10]  nearest positive food is right of head
        [11]  nearest poison is above head
        [12]  nearest poison is below head
        [13]  nearest poison is left of head
        [14]  nearest poison is right of head

    Backward compat: with n_silver=0 and n_poison=0, features [7-10] point to
    gold food (same as before) and features [11-14] are always 0. The network
    learns to ignore them, so the simple game still works.
    """
    hr, hc = env.snake[0]
    d = env.direction

    def ahead(direction: int) -> tuple[int, int]:
        dr, dc = _DELTA[direction]
        return (hr + dr, hc + dc)

    danger_straight = float(_is_dangerous(env, ahead(d)))
    danger_right    = float(_is_dangerous(env, ahead(_TURN_RIGHT[d])))
    danger_left     = float(_is_dangerous(env, ahead(_TURN_LEFT[d])))

    # Nearest positive food (gold or silver) — falls back to head if none exists
    pos_food = _nearest_food_of_type(env, ("gold", "silver")) or (hr, hc)
    fr, fc = pos_food

    # Nearest poison — direction features are zeroed when there is no poison
    poison_pos = _nearest_food_of_type(env, ("poison",))
    has_poison = poison_pos is not None
    pr, pc = poison_pos if has_poison else (hr, hc)

    return np.array(
        [
            danger_straight,
            danger_right,
            danger_left,
            float(d == UP),
            float(d == RIGHT),
            float(d == DOWN),
            float(d == LEFT),
            float(fr < hr),                      # positive food above
            float(fr > hr),                      # positive food below
            float(fc < hc),                      # positive food left
            float(fc > hc),                      # positive food right
            float(has_poison and pr < hr),        # poison above
            float(has_poison and pr > hr),        # poison below
            float(has_poison and pc < hc),        # poison left
            float(has_poison and pc > hc),        # poison right
        ],
        dtype=np.float32,
    )


def get_grid_state(env: SnakeEnv) -> np.ndarray:
    """Return the full grid as a 6-channel binary uint8 array (C, H, W).

    Using uint8 (values 0/1) instead of float32 reduces replay-buffer memory
    by 4× for CNN agents. Conversion to float32 happens at sample time inside
    ReplayBuffer.sample(), so networks always receive float32 tensors.

    Channels:
        0 = snake body
        1 = snake head
        2 = gold food
        3 = silver food
        4 = poison food
        5 = obstacle (static or dynamic)
    """
    obs = env._get_observation()  # (H, W) with values 0-6
    grid = np.zeros((6, env.height, env.width), dtype=np.uint8)
    grid[0] = obs == 1  # body
    grid[1] = obs == 2  # head
    grid[2] = obs == 3  # gold food
    grid[3] = obs == 4  # silver food
    grid[4] = obs == 5  # poison food
    grid[5] = obs == 6  # obstacle
    return grid


# ---------------------------------------------------------------------------
# Frame stacking
# ---------------------------------------------------------------------------


class FrameStack:
    """Stacks the last n_frames observations into a single state array.

    For the CNN agent: (6, H, W) per frame → (6*n_frames, H, W) stacked.
    For the MLP agent: (15,) per frame     → (15*n_frames,) concatenated.

    With n_frames=1 (default) the output is identical to a single frame,
    so all existing code paths are unaffected.
    """

    def __init__(self, n_frames: int, state_fn) -> None:
        self.n_frames = n_frames
        self.state_fn = state_fn
        self._frames: deque = deque(maxlen=n_frames)

    def reset(self, env: SnakeEnv) -> np.ndarray:
        """Call at the start of every episode to fill the buffer with the first frame."""
        obs = self.state_fn(env)
        self._frames.clear()
        for _ in range(self.n_frames):
            self._frames.append(obs)
        return self._stacked()

    def step(self, env: SnakeEnv) -> np.ndarray:
        """Call after every env.step() to push the new frame and return the stack."""
        self._frames.append(self.state_fn(env))
        return self._stacked()

    def _stacked(self) -> np.ndarray:
        if self.n_frames == 1:
            return self._frames[0]
        return np.concatenate(list(self._frames), axis=0)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """Circular numpy replay buffer.

    Compared to a deque-based buffer, numpy O(1) fancy-index sampling avoids
    the O(capacity) deque→list materialisation that random.sample() requires.
    Arrays are allocated lazily on the first push so the state shape does not
    need to be known at construction time.
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self._capacity = capacity
        self._pos = 0
        self._size = 0
        # Allocated on first push once we know state shape/dtype
        self._states: np.ndarray | None = None
        self._next_states: np.ndarray | None = None
        self._actions: np.ndarray | None = None
        self._rewards: np.ndarray | None = None
        self._dones: np.ndarray | None = None

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        if self._states is None:
            self._states = np.empty((self._capacity, *state.shape), dtype=state.dtype)
            self._next_states = np.empty_like(self._states)
            self._actions = np.empty(self._capacity, dtype=np.int64)
            self._rewards = np.empty(self._capacity, dtype=np.float32)
            self._dones = np.empty(self._capacity, dtype=np.float32)
        self._states[self._pos] = state
        self._next_states[self._pos] = next_state
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._dones[self._pos] = float(done)
        self._pos = (self._pos + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self._size, size=batch_size)
        states      = self._states[idxs]
        next_states = self._next_states[idxs]
        # CNN stores uint8 (0/1 binary grid) to save 4× memory; convert here.
        if states.dtype != np.float32:
            states      = states.astype(np.float32)
            next_states = next_states.astype(np.float32)
        return (
            states,
            self._actions[idxs],
            self._rewards[idxs],
            next_states,
            self._dones[idxs],
        )

    def __len__(self) -> int:
        return self._size


class _NStepBuffer:
    """Accumulates transitions and emits n-step compressed (s, a, G, s_n, done) tuples.

    G = r_0 + γ·r_1 + … + γ^(n-1)·r_{n-1}  (partial sum; bootstrap handled in learn())
    done = any terminal flag in the window (so bootstrap is skipped correctly)

    Usage: call push() every step; it feeds completed transitions into `replay`.
    When the episode ends call flush() to drain the remaining window.
    """

    def __init__(self, n: int, gamma: float, replay: ReplayBuffer) -> None:
        self.n = n
        self.gamma = gamma
        self.replay = replay
        self._window: deque = deque()  # unbounded; we manage size manually

    def push(self, state, action, reward, next_state, done) -> None:
        self._window.append((state, action, reward, next_state, done))
        # Emit one n-step transition whenever the window is full
        if len(self._window) >= self.n:
            self._emit()
        if done:
            # Flush remaining transitions (< n steps) at episode end
            while self._window:
                self._emit()

    def _emit(self) -> None:
        n = min(self.n, len(self._window))
        s0, a0 = self._window[0][0], self._window[0][1]
        G = sum(self.gamma ** i * self._window[i][2] for i in range(n))
        s_next = self._window[n - 1][3]
        # done=True if any transition in the window is terminal
        done_n = any(t[4] for t in list(self._window)[:n])
        self.replay.push(s0, a0, G, s_next, done_n)
        self._window.popleft()


# ---------------------------------------------------------------------------
# Base Agent
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, state: np.ndarray) -> int: ...

    @abstractmethod
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None: ...

    @abstractmethod
    def learn(self) -> float | None: ...

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


class _QNetwork(nn.Module):
    """MLP Q-network with optional dueling architecture."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: tuple[int, ...] = (256, 128),
        dueling: bool = False,
    ) -> None:
        super().__init__()
        self.dueling = dueling

        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        if dueling:
            self.value_head = nn.Linear(in_dim, 1)
            self.adv_head   = nn.Linear(in_dim, n_actions)
        else:
            self.out = nn.Linear(in_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.trunk(x)
        if self.dueling:
            v = self.value_head(x)               # (B, 1)
            a = self.adv_head(x)                 # (B, n_actions)
            return v + a - a.mean(dim=1, keepdim=True)
        return self.out(x)


class _CNNQNetwork(nn.Module):
    """CNN Q-network with optional dueling architecture.

    Operates on a stacked grid observation of shape (in_channels, H, W).
    With n_frames=1: in_channels=6. With n_frames=4: in_channels=24.
    """

    def __init__(
        self,
        height: int,
        width: int,
        n_actions: int,
        in_channels: int = 6,
        conv_channels: tuple[int, ...] = (32, 64),
        hidden: tuple[int, ...] = (512,),
        dueling: bool = False,
    ) -> None:
        super().__init__()
        self.dueling = dueling

        conv_layers: list[nn.Module] = []
        in_ch = in_channels
        for ch in conv_channels:
            conv_layers += [nn.Conv2d(in_ch, ch, kernel_size=3, padding=1), nn.ReLU()]
            in_ch = ch
        self.conv = nn.Sequential(*conv_layers)

        fc_layers: list[nn.Module] = []
        in_dim = in_ch * height * width
        for h in hidden:
            fc_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.fc_trunk = nn.Sequential(*fc_layers)

        if dueling:
            self.value_head = nn.Linear(in_dim, 1)
            self.adv_head   = nn.Linear(in_dim, n_actions)
        else:
            self.out = nn.Linear(in_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc_trunk(x)
        if self.dueling:
            v = self.value_head(x)
            a = self.adv_head(x)
            return v + a - a.mean(dim=1, keepdim=True)
        return self.out(x)


# ---------------------------------------------------------------------------
# Shared DQN logic
# ---------------------------------------------------------------------------

_OPTIMIZERS = {
    "adam": optim.Adam,
    "rmsprop": optim.RMSprop,
}


class _BaseDQNAgent(BaseAgent):
    """Shared DQN training logic. Subclasses provide the Q-network architecture."""

    N_ACTIONS = 4

    def __init__(
        self,
        q_net: nn.Module,
        lr: float,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        batch_size: int,
        buffer_capacity: int,
        target_update_freq: int,
        optimizer_name: str,
        device: torch.device,
        double_dqn: bool = False,
        grad_clip: float | None = 10.0,
        target_tau: float | None = None,
        n_step: int = 1,
    ) -> None:
        self.gamma = gamma
        self.n_step = n_step
        # gamma^n is used as the bootstrap discount in learn(); rewards in the
        # replay buffer already encode the n-step cumulative return G.
        self.gamma_n = gamma ** n_step
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        self.double_dqn = double_dqn
        self.grad_clip = grad_clip
        # target_tau: None = hard update every target_update_freq steps
        #             >0   = soft Polyak update every step (e.g. 0.005)
        self.target_tau = target_tau

        self.q_net = q_net
        self.target_net = copy.deepcopy(q_net)
        self.target_net.eval()

        optimizer_cls = _OPTIMIZERS.get(optimizer_name)
        if optimizer_cls is None:
            msg = f"Unknown optimizer '{optimizer_name}'. Choose from: {list(_OPTIMIZERS)}"
            raise ValueError(msg)
        self.optimizer = optimizer_cls(self.q_net.parameters(), lr=lr)

        self.buffer = ReplayBuffer(buffer_capacity)
        # n-step accumulator — wraps self.buffer; passthrough when n_step=1
        self._n_step_buf: _NStepBuffer | None = (
            _NStepBuffer(n_step, gamma, self.buffer) if n_step > 1 else None
        )
        self._steps = 0

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.N_ACTIONS)
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.q_net(t).argmax(dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        if self._n_step_buf is not None:
            self._n_step_buf.push(state, action, reward, next_state, done)
        else:
            self.buffer.push(state, action, reward, next_state, done)

    def learn(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # torch.as_tensor avoids an extra Python object vs torch.tensor()
        s  = torch.as_tensor(states,      dtype=torch.float32, device=self.device)
        a  = torch.as_tensor(actions,     dtype=torch.int64,   device=self.device)
        r  = torch.as_tensor(rewards,     dtype=torch.float32, device=self.device)
        s_ = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        d  = torch.as_tensor(dones,       dtype=torch.float32, device=self.device)

        q_values = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                # Online net selects the action, target net evaluates it
                best_actions = self.q_net(s_).argmax(dim=1)
                max_next_q = self.target_net(s_).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            else:
                max_next_q = self.target_net(s_).max(dim=1).values
            # r already encodes the n-step cumulative return G; gamma_n = gamma^n
            targets = r + self.gamma_n * max_next_q * (1.0 - d)

        # Huber loss is less sensitive to outlier Q-value targets than MSE
        loss = nn.functional.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        self._steps += 1
        if self.target_tau is not None:
            # Soft (Polyak) update every step: θ_target ← τ·θ_online + (1-τ)·θ_target
            for p_online, p_target in zip(self.q_net.parameters(), self.target_net.parameters(), strict=True):
                p_target.data.copy_(
                    self.target_tau * p_online.data + (1.0 - self.target_tau) * p_target.data,
                )
        elif self._steps % self.target_update_freq == 0:
            # Hard update
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path: str) -> None:
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "steps": self._steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self._steps = ckpt["steps"]


# ---------------------------------------------------------------------------
# Concrete agents
# ---------------------------------------------------------------------------


class DQNAgent(_BaseDQNAgent):
    """DQN agent with a fully-connected MLP on the 15-feature state vector.

    With n_frames=1 (default): input dim = 15.
    With n_frames=N:           input dim = 15 * N (last N feature vectors stacked).

    Backward compat: on the simple env (no silver/poison/obstacles) features
    [11-14] are always 0, so the agent behaves identically to the old 11-feature
    version (the extra zeros are ignored by the network).
    """

    BASE_STATE_DIM = 15

    def __init__(
        self,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        target_update_freq: int = 1_000,
        hidden: tuple[int, ...] = (256, 128),
        optimizer_name: str = "adam",
        device: str | None = None,
        n_frames: int = 1,
        double_dqn: bool = False,
        dueling: bool = False,
        grad_clip: float | None = 10.0,
        target_tau: float | None = None,
        n_step: int = 1,
    ) -> None:
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        state_dim = self.BASE_STATE_DIM * n_frames
        q_net = _QNetwork(state_dim, self.N_ACTIONS, hidden, dueling=dueling).to(dev)
        super().__init__(
            q_net=q_net,
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            batch_size=batch_size,
            buffer_capacity=buffer_capacity,
            target_update_freq=target_update_freq,
            optimizer_name=optimizer_name,
            device=dev,
            double_dqn=double_dqn,
            grad_clip=grad_clip,
            target_tau=target_tau,
            n_step=n_step,
        )


class CNNDQNAgent(_BaseDQNAgent):
    """DQN agent with a CNN on the stacked grid observation.

    With n_frames=1 (default): input channels = 6.
    With n_frames=N:           input channels = 6 * N.
    """

    BASE_CHANNELS = 6

    def __init__(
        self,
        height: int,
        width: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        target_update_freq: int = 1_000,
        conv_channels: tuple[int, ...] = (32, 64),
        hidden: tuple[int, ...] = (512,),
        optimizer_name: str = "adam",
        device: str | None = None,
        n_frames: int = 1,
        double_dqn: bool = False,
        dueling: bool = False,
        grad_clip: float | None = 10.0,
        target_tau: float | None = None,
        n_step: int = 1,
    ) -> None:
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        in_channels = self.BASE_CHANNELS * n_frames
        q_net = _CNNQNetwork(
            height, width, self.N_ACTIONS, in_channels, conv_channels, hidden, dueling=dueling
        ).to(dev)
        super().__init__(
            q_net=q_net,
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            batch_size=batch_size,
            buffer_capacity=buffer_capacity,
            target_update_freq=target_update_freq,
            optimizer_name=optimizer_name,
            device=dev,
            double_dqn=double_dqn,
            grad_clip=grad_clip,
            target_tau=target_tau,
            n_step=n_step,
        )
