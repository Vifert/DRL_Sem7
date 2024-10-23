import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from torch.optim.lr_scheduler import StepLR

def impute_missing_detection(last_box, frame_width, frame_height):
    if last_box is not None:
        adjustment = np.random.uniform(-0.02, 0.02, size=4)
        imputed_box = last_box + adjustment
        imputed_box[[0, 2]] = np.clip(imputed_box[[0, 2]], 0, 1)
        imputed_box[[1, 3]] = np.clip(imputed_box[[1, 3]], 0, 1)
    else:
        imputed_box = np.array([0.4, 0.4, 0.6, 0.6])  # Centered box
    return imputed_box

def process_video(video_path, yolo_model, max_frames=1000, confidence_threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bounding_boxes = []
    frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)
        boxes = results[0].boxes

        if len(boxes) > 0 and boxes[0].conf.item() > confidence_threshold:
            box = boxes[0].xyxy[0].cpu().numpy()
            box[[0, 2]] /= width
            box[[1, 3]] /= height
            bounding_boxes.append(box)
        else:
            imputed_box = impute_missing_detection(bounding_boxes[-1] if bounding_boxes else None, width, height)
            bounding_boxes.append(imputed_box)

        frame_count += 1

    cap.release()
    return np.array(bounding_boxes)

def prepare_sequences(bounding_boxes, sequence_length):
    sequences = []
    for i in range(len(bounding_boxes) - sequence_length):
        seq = bounding_boxes[i:i + sequence_length + 1]
        sequences.append(seq)
    return np.array(sequences)

class EnhancedTrajectoryPredictor(nn.Module):
    def __init__(self, input_size=4, hidden_size=256, num_layers=3):
        super(EnhancedTrajectoryPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_size, 4)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        x = self.relu(self.fc1(attn_out[:, -1, :]))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_trajectory_predictor(train_sequences, val_sequences, epochs=50, batch_size=64):
    model = EnhancedTrajectoryPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)  # Reduce LR by half every 30 epochs

    X_train = torch.FloatTensor(train_sequences[:, :-1])
    y_train = torch.FloatTensor(train_sequences[:, -1])
    X_val = torch.FloatTensor(val_sequences[:, :-1])
    y_val = torch.FloatTensor(val_sequences[:, -1])

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], "
                  f"Train Loss: {train_loss / len(train_loader):.4f}, "
                  f"Val Loss: {val_loss.item():.4f}")

    return model

class RefinedBoundingBoxEnv(gym.Env):
    def __init__(self, sequences, trajectory_predictor, sequence_length=10):
        super(RefinedBoundingBoxEnv, self).__init__()
        self.sequences = sequences
        self.trajectory_predictor = trajectory_predictor
        self.sequence_length = sequence_length

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(sequence_length * 4 + 12,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-0.05, high=0.05, shape=(4,), dtype=np.float32
        )

        self.current_sequence = None
        self.current_step = 0
        self.max_steps = None

    def reset(self):
        self.current_sequence_index = np.random.randint(0, len(self.sequences))
        self.full_sequence = self.sequences[self.current_sequence_index]
        self.current_step = 0
        self.max_steps = len(self.full_sequence) - self.sequence_length - 1

        self.current_sequence = self.full_sequence[self.current_step:self.current_step + self.sequence_length]
        return self._get_obs()

    def step(self, action):
        with torch.no_grad():
            pred_box = self.trajectory_predictor(
                torch.FloatTensor(self.current_sequence).unsqueeze(0)
            ).squeeze(0).numpy()

        adjusted_box = np.clip(pred_box + action, 0, 1)
        true_next_box = self.full_sequence[self.current_step + self.sequence_length]

        position_error = np.mean((adjusted_box[:2] - true_next_box[:2]) ** 2)
        size_error = np.mean((adjusted_box[2:] - true_next_box[2:]) ** 2)
        smoothness_penalty = np.mean((adjusted_box - self.current_sequence[-1]) ** 2)
        reward = -(position_error + 0.5 * size_error + 0.1 * smoothness_penalty)

        self.current_sequence = np.roll(self.current_sequence, -1, axis=0)
        self.current_sequence[-1] = true_next_box

        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        velocities = np.diff(self.current_sequence, axis=0)
        accelerations = np.diff(velocities, axis=0)
        return np.concatenate([
            self.current_sequence.flatten(),
            velocities[-1],
            accelerations[-1],
            self.current_sequence[-1] - self.current_sequence[0]
        ])

def visualize_predictions(video_path, trajectory_model, ppo_model, yolo_model, sequence_length=10, num_steps=400, output_path='output_video.mp4'):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frames = []
    for _ in range(sequence_length):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    if len(frames) < sequence_length:
        print("Not enough frames in video.")
        cap.release()
        out.release()
        return

    initial_bboxes = []
    for frame in frames:
        results = yolo_model(frame)
        boxes = results[0].boxes
        if len(boxes) > 0:
            box = boxes[0].xyxy[0].cpu().numpy()
            box[[0, 2]] /= width
            box[[1, 3]] /= height
            initial_bboxes.append(box)
        else:
            imputed_box = impute_missing_detection(initial_bboxes[-1] if initial_bboxes else None, width, height)
            initial_bboxes.append(imputed_box)

    current_sequence = np.array(initial_bboxes)
    velocities = np.diff(current_sequence, axis=0)
    accelerations = np.diff(velocities, axis=0)
    obs = np.concatenate([
        current_sequence.flatten(),
        velocities[-1],
        accelerations[-1],
        current_sequence[-1] - current_sequence[0]
    ])

    for frame_idx in range(num_steps):
        action, _ = ppo_model.predict(obs)

        with torch.no_grad():
            pred_box = trajectory_model(torch.FloatTensor(current_sequence).unsqueeze(0)).squeeze(0).numpy()
        adjusted_box = np.clip(pred_box + action, 0, 1)

        ret, frame = cap.read()
        if not ret:
            break

        pred_box_coords = (
            int(adjusted_box[0] * width), int(adjusted_box[1] * height),
            int(adjusted_box[2] * width), int(adjusted_box[3] * height)
        )
        cv2.rectangle(frame,
                      (pred_box_coords[0], pred_box_coords[1]),
                      (pred_box_coords[2], pred_box_coords[3]),
                      (0, 0, 255), 2)
        cv2.putText(frame, 'Predicted', (pred_box_coords[0], pred_box_coords[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        results = yolo_model(frame)
        boxes = results[0].boxes
        if len(boxes) > 0:
            actual_box = boxes[0].xyxy[0].cpu().numpy()
            cv2.rectangle(frame,
                          (int(actual_box[0]), int(actual_box[1])),
                          (int(actual_box[2]), int(actual_box[3])),
                          (0, 255, 0), 2)
            cv2.putText(frame, 'Actual', (int(actual_box[0]), int(actual_box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = adjusted_box
        velocities = np.diff(current_sequence, axis=0)
        accelerations = np.diff(velocities, axis=0)
        obs = np.concatenate([
            current_sequence.flatten(),
            velocities[-1],
            accelerations[-1],
            current_sequence[-1] - current_sequence[0]
        ])

    cap.release()
    out.release()
    print(f"Output video saved to {output_path}")

def main():
    yolo_model = YOLO('D:/drown3/kaggle/working/runs/detect/train/weights/best.pt')
    video_path = "D:/video0001-1534.mp4"

    max_frames = 1400  # Adjust as necessary
    bounding_boxes = process_video(video_path, yolo_model, max_frames)

    sequence_length = 10  # Increased from 1
    sequences = prepare_sequences(bounding_boxes, sequence_length)

    train_sequences, val_sequences = train_test_split(sequences, test_size=0.2, random_state=42)

    trajectory_model = train_trajectory_predictor(train_sequences, val_sequences, epochs=50, batch_size=64)

    env = RefinedBoundingBoxEnv(train_sequences, trajectory_model, sequence_length)

    ppo_model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64,
                    n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                    clip_range_vf=None, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                    use_sde=False, sde_sample_freq=-1, target_kl=None)

    ppo_model.learn(total_timesteps=50000)  # Increased from 10000

    mean_reward, std_reward = evaluate_policy(ppo_model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    output_video_path = 'D:/output_video.mp4'  # Replace with your desired output path
    visualize_predictions(video_path, trajectory_model, ppo_model, yolo_model,
                          sequence_length=sequence_length, num_steps=1400, output_path=output_video_path)

if __name__ == "__main__":
    main()