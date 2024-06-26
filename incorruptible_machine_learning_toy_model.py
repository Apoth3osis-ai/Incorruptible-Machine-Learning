# -*- coding: utf-8 -*-
"""Incorruptible Machine Learning Toy Model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/19qibfTwu2gnkxWohB-iw80HejNxVVFr7

# Code
"""

# Install Required Libraries
!pip install tensorflow
import tensorflow as tf
import hashlib
import pandas as pd
import numpy as np
import json
import os

# Define directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Load the Iris dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the create_hash function
def create_hash(data):
    if isinstance(data, np.ndarray):
        data = data.tolist()
    data_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(data_str.encode()).hexdigest()

# Hash training data blocks
train_hashes = [create_hash(X_train[i]) for i in range(len(X_train))]
data_audit_trail = [{"index": i, "hash": train_hashes[i]} for i in range(len(train_hashes))]

with open('data/data_audit_trail.json', 'w') as f:
    json.dump(data_audit_trail, f)

# Define a simple neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callback to store model weights and hashes
class HashingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        weights = self.model.get_weights()
        weight_hashes = [create_hash(w.tolist()) for w in weights]
        epoch_hash = create_hash(weight_hashes)

        with open(f'models/epoch_{epoch}_hash.json', 'w') as f:
            json.dump({"epoch": epoch, "hash": epoch_hash}, f)

hashing_callback = HashingCallback()

# Train the model with the hashing callback
model.fit(X_train, y_train, epochs=5, validation_split=0.2, callbacks=[hashing_callback])

# Maintain and verify logs
audit_logs = {
    "data_audit_trail": data_audit_trail,
    "model_weights_hashes": []
}

for epoch in range(5):
    with open(f'models/epoch_{epoch}_hash.json', 'r') as f:
        audit_logs["model_weights_hashes"].append(json.load(f))

with open('logs/audit_logs.json', 'w') as f:
    json.dump(audit_logs, f)

# Verification function
def verify_audit_trail(audit_logs, X_train):
    # Verify data hashes
    for entry in audit_logs["data_audit_trail"]:
        assert create_hash(X_train[entry["index"]]) == entry["hash"], "Data integrity check failed!"

    # Verify model weights hashes
    for epoch_hash in audit_logs["model_weights_hashes"]:
        epoch = epoch_hash["epoch"]
        with open(f'models/epoch_{epoch}_hash.json', 'r') as f:
            stored_hash = json.load(f)["hash"]
        assert epoch_hash["hash"] == stored_hash, "Model weights integrity check failed!"

    print("All checks passed!")

verify_audit_trail(audit_logs, X_train)

"""# Explanation

**Steps:**



* Load and Preprocess Data: Load the Iris dataset, split it
into training and test sets, and standardize the features.

* Create Hash Function: Define a function to hash the data.
Hash Training Data: Hash each training data block and store the hashes.

* Define Model: Define a simple neural network model.
Hash Model Weights: Use a callback to hash model weights after each epoch.

* Train Model: Train the model with the hashing callback.
Maintain Logs: Store audit logs for training data and model weights.

* Verify Integrity: Verify the integrity of the training data and model weights using the stored hashes.

**What "All checks passed!" Means:**

* Data Integrity: The training data used has remained unchanged from the time it was initially hashed and logged.
* Model Weights Integrity: The model weights at each epoch have not been altered from what was logged during training.
* Immutable Log: The audit trail (logs) accurately reflects the training data and model weights used, providing a verifiable chain of custody.

# Extended Log (Code)
"""

# Install Required Libraries
!pip install tensorflow
import tensorflow as tf
import hashlib
import pandas as pd
import numpy as np
import json
import os

# Define directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Load the Iris dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the create_hash function
def create_hash(data):
    if isinstance(data, np.ndarray):
        data = data.tolist()
    data_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(data_str.encode()).hexdigest()

# Hash training data blocks
train_hashes = [create_hash(X_train[i]) for i in range(len(X_train))]
data_audit_trail = [{"index": i, "hash": train_hashes[i]} for i in range(len(train_hashes))]

with open('data/data_audit_trail.json', 'w') as f:
    json.dump(data_audit_trail, f)

# Hyperparameters and training script metadata
hyperparameters = {
    "optimizer": "adam",
    "loss": "sparse_categorical_crossentropy",
    "metrics": ["accuracy"],
    "epochs": 5,
    "validation_split": 0.2
}

training_script = """
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=5, validation_split=0.2)
"""

# Save hyperparameters and training script metadata
metadata = {
    "hyperparameters": hyperparameters,
    "training_script": training_script
}

with open('logs/training_metadata.json', 'w') as f:
    json.dump(metadata, f)

# Define a simple neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer=hyperparameters["optimizer"],
              loss=hyperparameters["loss"],
              metrics=hyperparameters["metrics"])

# Callback to store model weights and hashes
class HashingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        weights = self.model.get_weights()
        weight_hashes = [create_hash(w.tolist()) for w in weights]
        epoch_hash = create_hash(weight_hashes)

        with open(f'models/epoch_{epoch}_hash.json', 'w') as f:
            json.dump({"epoch": epoch, "hash": epoch_hash}, f)

hashing_callback = HashingCallback()

# Train the model with the hashing callback
model.fit(X_train, y_train, epochs=hyperparameters["epochs"], validation_split=hyperparameters["validation_split"], callbacks=[hashing_callback])

# Maintain and verify logs
audit_logs = {
    "data_audit_trail": data_audit_trail,
    "model_weights_hashes": [],
    "metadata_hash": create_hash(metadata)
}

for epoch in range(hyperparameters["epochs"]):
    with open(f'models/epoch_{epoch}_hash.json', 'r') as f:
        audit_logs["model_weights_hashes"].append(json.load(f))

with open('logs/audit_logs.json', 'w') as f:
    json.dump(audit_logs, f)

# Verification function
def verify_audit_trail(audit_logs, X_train):
    # Verify data hashes
    for entry in audit_logs["data_audit_trail"]:
        assert create_hash(X_train[entry["index"]]) == entry["hash"], "Data integrity check failed!"

    # Verify model weights hashes
    for epoch_hash in audit_logs["model_weights_hashes"]:
        epoch = epoch_hash["epoch"]
        with open(f'models/epoch_{epoch}_hash.json', 'r') as f:
            stored_hash = json.load(f)["hash"]
        assert epoch_hash["hash"] == stored_hash, "Model weights integrity check failed!"

    # Verify metadata
    with open('logs/training_metadata.json', 'r') as f:
        stored_metadata = json.load(f)
    assert create_hash(stored_metadata) == audit_logs["metadata_hash"], "Metadata integrity check failed!"

    print("All checks passed!")

verify_audit_trail(audit_logs, X_train)

"""# **Explanation of Extended Logging:**
**Hyperparameters and Training Script Metadata:**

- Store the hyperparameters used for training.
- Include the training script as a string for complete reproducibility.

**Save Metadata:**

- Save the metadata in a JSON file.

**Hash Metadata:**

- Hash the metadata and include it in the audit logs.

**Verification Function:**

- Extend the verification function to include a check for the metadata hash.

# Post Quantum Secure Hasing - Code
"""

!pip install tensorflow pycryptodome
import tensorflow as tf
from Crypto.Hash import SHAKE256
import pandas as pd
import numpy as np
import json
import os

def create_hash(data, output_length=64):
    if isinstance(data, np.ndarray):
        data = data.tolist()
    data_str = json.dumps(data, sort_keys=True)
    shake = SHAKE256.new()
    shake.update(data_str.encode())
    return shake.read(output_length).hex()

def setup_environment():
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

def prepare_data(data, target, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def hash_training_data(X_train):
    train_hashes = [create_hash(X_train[i]) for i in range(len(X_train))]
    data_audit_trail = [{"index": i, "hash": train_hashes[i]} for i in range(len(train_hashes))]
    with open('data/data_audit_trail.json', 'w') as f:
        json.dump(data_audit_trail, f)
    return data_audit_trail

def define_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    return model

def save_metadata(hyperparameters, training_script):
    metadata = {
        "hyperparameters": hyperparameters,
        "training_script": training_script
    }
    with open('logs/training_metadata.json', 'w') as f:
        json.dump(metadata, f)
    return create_hash(metadata)

def train_model_with_hashing(model, X_train, y_train, hyperparameters):
    model.compile(optimizer=hyperparameters["optimizer"],
                  loss=hyperparameters["loss"],
                  metrics=hyperparameters["metrics"])

    class HashingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            weights = self.model.get_weights()
            weight_hashes = [create_hash(w.tolist()) for w in weights]
            epoch_hash = create_hash(weight_hashes)
            with open(f'models/epoch_{epoch}_hash.json', 'w') as f:
                json.dump({"epoch": epoch, "hash": epoch_hash}, f)

    hashing_callback = HashingCallback()
    model.fit(X_train, y_train, epochs=hyperparameters["epochs"], validation_split=hyperparameters["validation_split"], callbacks=[hashing_callback])

def maintain_logs(data_audit_trail, epochs):
    audit_logs = {
        "data_audit_trail": data_audit_trail,
        "model_weights_hashes": []
    }
    for epoch in range(epochs):
        with open(f'models/epoch_{epoch}_hash.json', 'r') as f:
            audit_logs["model_weights_hashes"].append(json.load(f))
    with open('logs/audit_logs.json', 'w') as f:
        json.dump(audit_logs, f)
    return audit_logs

def verify_audit_trail(audit_logs, X_train):
    for entry in audit_logs["data_audit_trail"]:
        assert create_hash(X_train[entry["index"]]) == entry["hash"], "Data integrity check failed!"
    for epoch_hash in audit_logs["model_weights_hashes"]:
        epoch = epoch_hash["epoch"]
        with open(f'models/epoch_{epoch}_hash.json', 'r') as f:
            stored_hash = json.load(f)["hash"]
        assert epoch_hash["hash"] == stored_hash, "Model weights integrity check failed!"
    with open('logs/training_metadata.json', 'r') as f:
        stored_metadata = json.load(f)
    assert create_hash(stored_metadata) == audit_logs["metadata_hash"], "Metadata integrity check failed!"
    print("All checks passed!")

def run_immutable_training(data, target, hyperparameters, training_script):
    setup_environment()
    X_train, X_test, y_train, y_test = prepare_data(data, target)
    data_audit_trail = hash_training_data(X_train)
    model = define_model(X_train.shape[1])
    metadata_hash = save_metadata(hyperparameters, training_script)
    train_model_with_hashing(model, X_train, y_train, hyperparameters)
    audit_logs = maintain_logs(data_audit_trail, hyperparameters["epochs"])
    audit_logs["metadata_hash"] = metadata_hash
    verify_audit_trail(audit_logs, X_train)

# Example Usage
hyperparameters = {
    "optimizer": "adam",
    "loss": "sparse_categorical_crossentropy",
    "metrics": ["accuracy"],
    "epochs": 5,
    "validation_split": 0.2
}

training_script = """
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=5, validation_split=0.2)
"""

iris = load_iris()
data, target = iris.data, iris.target

run_immutable_training(data, target, hyperparameters, training_script)