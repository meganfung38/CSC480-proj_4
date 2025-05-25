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
    """rotates a set of 2D points CCW by angle theta"""
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return pts @ r

def _make_box_pts(x, y, yaw, w, h):
    """compute the 4 corner points of a rotated rectangle (bounding box):
    - centered at (x, y)
    - rotated by yaw
    - width w (perpendicular to yaw), height h (along yaw)
    """
    hx, hy = w / 2, h / 2
    box = np.array([[-hx, -hy], [-hx, hy], [hx, hy], [hx, -hy]])
    box = _rotation(box, yaw)
    box += (x, y)
    return box

def _make_spaceship(pos, yaw, scale, l2w, t2l):
    """generates a triangular spaceship shape and its bounding box label.
    parameters:
    - pos-- center (x, y)
    - yaw-- orientation
    - scale-- overall size
    - 12w-- length to width ration
    - t2l-- tip to tip length ratio
    """
    dim_x, dim_y = scale, scale * l2w
    pts = np.array([
        (0, dim_y),  # tip
        (-dim_x / 2, 0),  # left base
        (0, dim_y * t2l),  # center between tip and base
        (dim_x / 2, 0),  # right base
    ])
    pts[:, 1] -= dim_y / 2  # center vertically
    pts = _rotation(pts, yaw)
    pts += pos
    return pts, np.array([*pos, yaw, dim_x, dim_y])



# === DATA ===
def make_data(image_size=200, noise_level=0.0, salt_pepper_prob=0.0, no_lines=0):
    """create a synthetic training image and label.
    - draws a spaceship at a random position, size, and orientation
    - adds optional noise
    - returns:
      - img: 200 by 200 image with spaceship perimeter drawn
      - label: [x, y, yaw, width, height]
    """
    img = np.zeros((image_size, image_size))  # blank image
    pos = np.random.randint(20, image_size - 20, size=2)  # random center
    yaw = np.random.rand() * 2 * np.pi  # random orientation

    # vary the ship scale and aspect ratios
    scale = np.random.uniform(20, 36)       # base size
    l2w = np.random.uniform(1.2, 2.0)        # length-to-width
    t2l = np.random.uniform(0.2, 0.5)        # yip-to-length

    pts, label = _make_spaceship(pos, yaw, scale, l2w, t2l)  # getting shape and label

    # draw spaceship perimeter into image
    rr, cc = polygon_perimeter(pts[:, 0], pts[:, 1])
    valid = (rr >= 0) & (rr < image_size) & (cc >= 0) & (cc < image_size)
    img[rr[valid], cc[valid]] = 1.0

    # add Gaussian noise
    if noise_level > 0:
        img += np.random.normal(0.0, noise_level, img.shape)

    # salt and pepper
    if salt_pepper_prob > 0:
        rand = np.random.rand(*img.shape)
        img[rand < (salt_pepper_prob / 2)] = 1.0
        img[rand > (1 - salt_pepper_prob / 2)] = 0.0

    # line noise
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



# === METRIC ===
def score_iou(ypred, ytrue):
    """calculate intersection over union between predicted and true boxes.
    returns:
    - 0-- if prediction is invalid (contains NaNs)
    """
    if np.isnan(ypred).any() or np.isnan(ytrue).any():
        return 0.0
    p = Polygon(_make_box_pts(*ypred))
    t = Polygon(_make_box_pts(*ytrue))
    return p.intersection(t).area / p.union(t).area



# === BATCH CREATION ===
def make_batch(batch_size, image_size=200):
    """generate a batch of training data (images and labels)"""
    imgs, labels = zip(*[make_data(image_size) for _ in range(batch_size)])
    return np.stack(imgs), np.stack(labels)



# === TRAIN ===
def train_model(batch_size=64, epochs=10, image_size=200, model_path="model.h5"):
    """train CNN model:
    - generator-- feeds synthetic batches
    - saves trained model to disk
    """
    model = gen_model(image_size)
    model.compile(optimizer=Adam(), loss=MeanSquaredError())
    model.summary()

    def generator():
        while True:
            yield make_batch(batch_size, image_size)

    model.fit(generator(), steps_per_epoch=100, epochs=epochs)
    model.save(model_path)
    print(f"✅ Model saved to {model_path}")



# === EVAL ===
def evaluate_model(model_path="model.h5", image_size=200):
    """evaluation by computing IoU on 100 test samples"""
    model = keras.models.load_model(model_path)
    ious = []

    for _ in tqdm(range(100)):
        img, label = make_data(image_size=image_size)
        pred = np.squeeze(model.predict(img[None]))
        ious.append(score_iou(pred, label))

    ious = np.array(ious)
    ap = (ious > 0.7).mean()
    print(f"📈 AP@0.7 = {ap:.4f}")



# === VISUALIZATION ===
def show_examples(model, image_size=200):
    """show 3 side by side examples of model predictions vs ground truth"""
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
def gen_model(image_size=200):
    """CNN architecture
    input:
    - (200, 200) image
    output:
    - tf.keras model-- [x, y, yaw, width, height]
    """
    model = Sequential()
    model.add(Reshape((image_size, image_size, 1), input_shape=(image_size, image_size)))
    #build your model here

    model.add(Flatten())
    model.add(Dense(5))
    return model


# === EXECUTION BLOCK ===
train_model(epochs=10)  # train model and save it

# Load the saved model
model = keras.models.load_model("model.h5")  # load trained model

# Evaluate on 100 test samples
evaluate_model()

# Show true vs. predicted bounding boxes
show_examples(model)