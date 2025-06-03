import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon_perimeter
from shapely.geometry import Polygon
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Activation, Reshape, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow import keras
from tqdm import tqdm


# === GEOMETRY ===
def _rotation(pts, theta):
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return pts @ r

def _make_box_pts(x, y, yaw, w, h):
    hx, hy = w / 2, h / 2
    box = np.array([[-hx, -hy], [-hx, hy], [hx, hy], [hx, -hy]])
    #box = _rotation(box, yaw)
    box += (x, y)
    return box

def _make_spaceship(pos, yaw, scale, l2w, t2l):
    dim_x, dim_y = scale, scale * l2w
    pts = np.array([
        (0, dim_y),
        (-dim_x / 2, 0),
        (0, dim_y * t2l),
        (dim_x / 2, 0),
    ])
    pts[:, 1] -= dim_y / 2
    #pts = _rotation(pts, yaw)
    pts += pos
    return pts, np.array([*pos, yaw, dim_x, dim_y])


# === DATA ===
def make_data(image_size=200, noise_level=0.0, salt_pepper_prob=0.0, no_lines=0):
    img = np.zeros((image_size, image_size))
    pos = np.random.randint(20, image_size - 20, size=2)
    yaw = 1

    # Vary the ship scale and aspect ratios
    scale = 25     # Base size
    l2w = 1   # Length-to-width
    t2l = 1        # Tip-to-length

    pts, label = _make_spaceship(pos, yaw, scale, l2w, t2l)

    rr, cc = polygon_perimeter(pts[:, 0], pts[:, 1])
    valid = (rr >= 0) & (rr < image_size) & (cc >= 0) & (cc < image_size)
    img[rr[valid], cc[valid]] = 1.0

    # Add Gaussian noise
    if noise_level > 0:
        img += np.random.normal(0.0, noise_level, img.shape)

    # Salt and pepper
    if salt_pepper_prob > 0:
        rand = np.random.rand(*img.shape)
        img[rand < (salt_pepper_prob / 2)] = 1.0
        img[rand > (1 - salt_pepper_prob / 2)] = 0.0

    # Line noise
    for _ in range(no_lines):
        x1, y1 = np.random.randint(0, image_size, 2)
        x2, y2 = np.random.randint(0, image_size, 2)
        num_points = max(abs(x2 - x1), abs(y2 - y1))
        xs = np.linspace(x1, x2, num_points).astype(np.int32)
        ys = np.linspace(y1, y2, num_points).astype(np.int32)
        xs = np.clip(xs, 0, image_size - 1)
        ys = np.clip(ys, 0, image_size - 1)
        img[ys, xs] = 1.0

    img = np.clip(img, 0.0, 1.0)

    return img.T, label

def score_iou(ypred, ytrue):
    if np.isnan(ypred).any() or np.isnan(ytrue).any():
        return 0.0
    p = Polygon(_make_box_pts(*ypred))
    t = Polygon(_make_box_pts(*ytrue))
    return p.intersection(t).area / p.union(t).area

def make_batch(batch_size, image_size=200):
    imgs, labels = zip(*[make_data(image_size) for _ in range(batch_size)])
    return np.stack(imgs), np.stack(labels)


# === TRAIN ===
def train_model(batch_size=64, epochs=10, image_size=200, model_path="model.h5"):
    model = gen_model(image_size)
    model.compile(optimizer=Adam(), loss=MeanSquaredError())
    model.summary()

    def generator():
        while True:
            yield make_batch(batch_size, image_size)

    model.fit(generator(), steps_per_epoch=100, epochs=epochs)
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")

def evaluate_model(model_path="model.h5", image_size=200, distance_threshold=10.0):
    model = keras.models.load_model(model_path)
    distances = []

    for _ in tqdm(range(100)):
        img, label = make_data(image_size=image_size)
        pred = np.squeeze(model.predict(img[None]))

        # Extract x and y from prediction and label
        x_pred, y_pred = pred[0], pred[1]
        x_true, y_true = label[0], label[1]

        # Compute Euclidean distance between predicted and true center
        dist = np.sqrt((x_pred - x_true)**2 + (y_pred - y_true)**2)
        distances.append(dist)

    distances = np.array(distances)
    accuracy = (distances <= distance_threshold).mean()
    print(f"📏 Center Accuracy @ {distance_threshold} pixels = {accuracy:.4f}")

def show_examples(model, image_size=200):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for i in range(3):
        img, label = make_data(image_size=image_size)
        pred = np.squeeze(model.predict(img[None]))

        ax[i].imshow(img, cmap="gray")
        ax[i].set_title("Red = True, Blue = Pred")

        # Draw ground truth in red
        if not np.isnan(label).any():
            pts_true = _make_box_pts(*label)
            ax[i].plot(pts_true[:, 0], pts_true[:, 1], c="red")
            ax[i].scatter(label[0], label[1], c="red")

        # Draw prediction in blue
        if not np.isnan(pred).any():
            pts_pred = _make_box_pts(*pred)
            ax[i].plot(pts_pred[:, 0], pts_pred[:, 1], c="blue", linestyle='--')
            ax[i].scatter(pred[0], pred[1], c="blue", marker='x')

    plt.tight_layout()
    plt.show()


# === MODEL ===
###This is you only task
def gen_model(image_size=200):
    model = Sequential()

    # add channel dimension: (200, 200) -> (200, 200, 1)
    model.add(Reshape((image_size, image_size, 1), input_shape=(image_size, image_size)))

    # layer 1
    model.add(Conv2D(32, kernel_size=5, strides=1, padding='same'))  # applies 32 5x5 filters
    model.add(BatchNormalization())  # normalize activation
    model.add(Activation('relu'))  # learn complex patterns
    model.add(MaxPool2D(pool_size=2))  # downsample: 200x200 -> 100x100

    # layer 2
    model.add(Conv2D(64, kernel_size=3, strides=1, padding='same'))  # applies 64 3x3 filters
    model.add(BatchNormalization())  # normalize activation
    model.add(Activation('relu'))  # learn complex patterns
    model.add(MaxPool2D(pool_size=2))  # downsample: 100x100 -> 50x50

    # layer 3
    model.add(Conv2D(128, kernel_size=3, strides=1, padding='same'))  # applies 128 3x3 filters
    model.add(BatchNormalization())  # normalize activation
    model.add(Activation('relu'))  # learn complex patterns
    model.add(MaxPool2D(pool_size=2))  # downsample: 50x50 -> 25x25

    model.add(Flatten())  # flatten into vector of length 25 * 25 * 128 = 80k
    model.add(Dense(128, activation='relu'))  # dense layer with 128 units for feature extraction
    model.add(Dense(64, activation='relu'))  # dense layer with 64 units for representation
    model.add(Dense(5))  # dense layer outputs [x, y, yaw, width, height]
    return model

# === EXECUTION BLOCK ===
train_model(epochs=10)

# Load the saved model
model = keras.models.load_model("model.h5")

# Evaluate on 100 test samples
evaluate_model()

# Show true vs. predicted bounding boxes
show_examples(model)

