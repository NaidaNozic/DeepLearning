# %% [markdown]
# # Assignment 2

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

print('seaborn:', sns.__version__)
print('pandas:', pd.__version__)
print('numpy:', np.__version__)
print('torch:', torch.__version__)

# Define a function to set the seed
def set_seed(seed):
    """
    Sets the seed for reproducibility using numpy and torch.
    """
    # Set seed for numpy
    np.random.seed(seed)
    print(f"Numpy random seed set to: {seed}")
    
    # Set seed for torch (CPU and GPU)
    torch.manual_seed(seed)  # For CPU
    torch.use_deterministic_algorithms(mode=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # For current GPU
        torch.cuda.manual_seed_all(seed)  # For all GPUs
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Torch random seed set to: {seed}")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed) 

generator = torch.Generator()
generator.manual_seed(0)

# Use the function
seed = 42
set_seed(seed)

# %% [markdown]
# ## Task a)

# %%
set_seed(seed)
# Splitting the data into train and test set
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=302)

columns=["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Lat", "Long", "MedHouseVal"]
data = pd.DataFrame(data=np.column_stack((X_train, y_train)), columns=columns)

print("Size of the data:")
print(data.shape)

data.head()


# %%
print("Structure of the data:")
print(data.describe())

# %%
print(data.info())

# %%
# Plotting feature distributions
def plot_feature_distributions(data, title):
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(data.columns, 1):
        plt.subplot(3, 3, i)
        sns.histplot(data[column], kde=True, bins=30)
        plt.title(column)
    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=16)
    plt.show()

plot_feature_distributions(data.iloc[:, :-1], "Feature Distributions in Dataset")

# %%
# Scatter plot to observe relationships between features and the target variable.
scatter_matrix = pd.plotting.scatter_matrix(
    data,
    alpha=0.25,
    figsize=(16, 16),
    diagonal="kde",
    marker="o"
)
plt.suptitle("Feature Relationships and Distributions", fontsize=18, y=1.02)
plt.show()

# %%
features = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Lat", "Long"]
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(30, 20))

for index, (axis, feature) in enumerate(zip(axes.flat, features)):
    feature_data = X[:, index] 
    fit_params = np.polyfit(feature_data, y, 1)
    linear_model = np.poly1d(fit_params)

    axis.scatter(feature_data, y, color='blue', label='Data points')
    axis.plot(feature_data, linear_model(feature_data), 'r--', label='Fitted line')

    axis.set_title(f"{feature} vs Median House Value")
    axis.set_xlabel(feature)
    axis.set_ylabel('Median House Value')
    axis.legend()

plt.tight_layout()
plt.show()

# %%
# Missing values count
print("Number of missing values")
print("------------------------------")
missing_values = data.isnull().sum()
missing_percentage = (missing_values / len(data)) * 100
missing_info = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage
})
print(missing_info)

print("\nNumber of unique values")
print("------------------------------")
unique_values = data.nunique()
print(unique_values)

# %%
print("Correlation matrix for features")
print("------------------------------")
correlation_matrix = data.corr()
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# %%
print("Number of outliers:")
print("------------------------------")
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

# Define outliers as values outside 1.5*IQR
outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))

outliers_count = outliers.sum()
print(outliers_count)

# %%
set_seed(seed)
# Preprocessing
# Normalization
scaler = StandardScaler()
x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)

# Split X_train to validation and train data
X_new_train, X_val, y_new_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=302)

# %% [markdown]
# ## Task b)

# %%
set_seed(seed)
# Convert to PyTorch tensors
X_new_train = torch.tensor(X_new_train).float()
y_new_train = torch.tensor(y_new_train).view(-1, 1).float()

X_val = torch.tensor(X_val).float()
y_val = torch.tensor(y_val).view(-1, 1).float()

X_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test).view(-1, 1).float()

# %%
set_seed(seed)
generator = torch.Generator()
generator.manual_seed(0)

def train_with_batch_size(model_fn, batch_size, num_epochs=200, patience=10):
    # Adjust learning rate based on batch size
    base_lr = 0.0001
    adjusted_lr = base_lr * (batch_size ** 0.5)

    train_dataset = TensorDataset(X_new_train, y_new_train)
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,worker_init_fn=seed_worker,generator=generator)

    val_dataset = TensorDataset(X_val, y_val)
    val_iter = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,worker_init_fn=seed_worker,generator=generator)

    model = model_fn()
    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=adjusted_lr)
    
    train_losses = []
    val_losses = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_train_loss = None 
    epochs_no_improve = 0
    early_stop = False
    
    for epoch in range(num_epochs):
        if early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

        running_train_loss = 0.0
        model.train()
        
        # Training loop
        for inputs, labels in train_iter:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        
        # Validation loop
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_iter:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
        
        # Calculate averate train and validation losses
        avg_train_loss = running_train_loss / len(train_iter)
        avg_val_loss = running_val_loss / len(val_iter)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_train_loss = avg_train_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            early_stop = True
    
    # Evaluate the model on test data
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_rmse = np.sqrt(mean_squared_error(test_outputs.numpy(), y_test.numpy()))

    return best_val_loss, test_rmse, best_train_loss

set_seed(seed)
# We will use a fixed model to see if its loss changes with different batch sizes

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0, std=0.01)  # Manual initialization of weights
        m.bias.data.fill_(0.0001)  # Bias initialization

def create_model():
    model = nn.Sequential(
        nn.Linear(8, 100),
        nn.ReLU(),
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    
    # Apply custom weight initialization
    model.apply(init_weights)
    
    return model

# Batch sizes to test
batch_sizes = [16, 32, 50, 64, 128, 256]
results = []

# Running experiments on every batch size
for batch_size in batch_sizes:
    print(f"Testing batch size: {batch_size}")
    val_loss, test_rmse, train_loss = train_with_batch_size(create_model, batch_size)
    results.append({'Batch Size': batch_size,'Train Loss': train_loss, 'Validation Loss': val_loss, 'Test RMSE': test_rmse })
    print(f"Batch Size: {batch_size}, Train Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f}, Test RMSE: {test_rmse:.3f}")

results_df = pd.DataFrame(results)
print(results_df)

best_result = results_df.loc[results_df['Validation Loss'].idxmin()]
print(f"Best Batch Size: {best_result['Batch Size']} with "
      f"Validation Loss: {best_result['Validation Loss']:.3f}, "
      f"Test RMSE: {best_result['Test RMSE']:.3f}, "
      f"Train Loss: {best_result['Train Loss']:.3f}")

# %% [markdown]
# ### We will choose 50 as the batch size since it provided the lowest Validation Loss and Test RMSE.

# %%
set_seed(seed)
generator = torch.Generator()
generator.manual_seed(0)

batch_size = 50
train_dataset = TensorDataset(X_new_train, y_new_train)
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    worker_init_fn=seed_worker,
    generator=generator)

val_dataset = TensorDataset(X_val, y_val)
val_iter = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    worker_init_fn=seed_worker,
    generator=generator)

# Architectures to test
architectures = [
    [8, 50, 1],
    [8, 100, 50, 1],
    [8, 64, 32, 16, 1],
    [8, 100, 50, 10, 1],
    [8, 32, 32, 16, 1]
]

# Function to create the model
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0, std=0.01)  # Manual initialization of weights and bias for reproducability
        m.bias.data.fill_(0.0001)

def create_model(architecture):
    layers = []
    for i in range(len(architecture) - 1):
        layer = nn.Linear(architecture[i], architecture[i + 1])
        layers.append(layer)
        if i < len(architecture) - 2:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    model.apply(init_weights)   
    return model

# Loss function
criterion = nn.MSELoss()

# Training function with early stopping and validation loss tracking
def train(model_inp, num_epochs=200, patience=40, print_interval=10):
    adjusted_lr = 0.0001 * (batch_size ** 0.5)
    optimizer = torch.optim.RMSprop(model_inp.parameters(), lr=adjusted_lr)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_train_loss = None
    best_epoch = None
    epochs_without_improvement = 0  # Counter for early stopping
    
    # Loop through epochs
    for epoch in range(num_epochs):
        running_train_loss = 0.0
        model_inp.train()
        
        # Training loop
        for inputs, labels in train_iter:
            outputs = model_inp(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        
        # Validation loop
        model_inp.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_iter:
                outputs = model_inp(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
        
        # Calculate the average losses
        avg_train_loss = running_train_loss / len(train_iter)
        avg_val_loss = running_val_loss / len(val_iter)
        
        # Store the losses
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # If validation loss improves, save model and update best loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_train_loss = avg_train_loss
            best_epoch = epoch
            epochs_without_improvement = 0  # Reset counter if validation loss improves
            
        else:
            epochs_without_improvement += 1
        
        # Early stopping condition
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
            break

        # Print the best validation loss every `print_interval` epochs
        if (epoch + 1) % print_interval == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {avg_train_loss:.3f}, Validation Loss: {avg_val_loss:.3f}")
            print(f"Best Validation Loss: {best_val_loss:.3f} at Epoch {best_epoch + 1}")

    return train_losses, val_losses, best_train_loss, best_val_loss, best_epoch

model_dir = "models"

train_val_loss_data = []
best_loss_data = []

for idx, architecture in enumerate(architectures):
    print(f"\nTraining model {idx + 1} with architecture: {architecture}")

    # Create model
    model = create_model(architecture)
    model.train()
    
    # Train the model
    model_path = model_dir + f"model_architecture_{idx + 1}.pth"
    train_losses, val_losses, best_train_loss, best_val_loss_at_epoch, best_epoch = train(model, num_epochs=200)
    
    # Collect the final train and validation loss
    for epoch in range(len(train_losses)):
        train_val_loss_data.append({
            'Model': f"Model_{idx + 1}",
            'Epoch': epoch + 1,  # Epoch starts from 1, not 0
            'Train Loss': train_losses[epoch],  # Loss for the current epoch
            'Validation Loss': val_losses[epoch]  # Validation loss for the current epoch
        })

    # Collect the best train loss, best validation loss, and best epoch
    best_loss_data.append({
        'Model': f"Model_{idx + 1}",
        'Best Train Loss': best_train_loss,
        'Best Validation Loss': best_val_loss_at_epoch,
        'Best Epoch': best_epoch
    })

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        err = np.sqrt(mean_squared_error(outputs.numpy(), y_test.numpy()))
        print(f"Model {idx + 1} RMSE on test set: {err:.3f}")

train_val_loss_summary = pd.DataFrame(train_val_loss_data)
best_loss_summary = pd.DataFrame(best_loss_data)

# Display both summaries
print("\nTraining and Validation Loss Summary:")
print(train_val_loss_summary)

print("\nBest Loss and Epoch Summary:")
print(best_loss_summary)

# %% [markdown]
# ### Chosen Architecture: `[8, 64, 32, 16, 1]` due to lowest validation loss
# This architecture represents a feedforward neural network with the following layers:
# 
# - **Input Layer:** 8 neurons
# - **Hidden Layer 1:** 100 neurons
# - **Hidden Layer 2:** 50 neurons
# - **Hidden Layer 3:** 10 neurons
# - **Output Layer:** 1 neuron
# 
# This structure defines the number of neurons in each layer, from the input to the output, with three hidden layers in between.
# 
# It was chosen due to the lowest validation loss set of 0.279
# 
# For the loss function we have used Mean Squared Error (MSE). The learning rate is adjusted depending on the batch size as follows: 0.0001 * (batch_size ** 0.5).
# 
# For the activation function we have utilized ReLU. However, the output layer does not have an activation function, since we expect a continuous value in the regression task.

# %% [markdown]
# ### Task c)

# %%
set_seed(seed)
generator = torch.Generator()
generator.manual_seed(0)

# **Batch Size**
batch_size = 50

# **Number of Epochs**
num_epochs = 200

# **Architecture**
architecture = [8, 64, 32, 16, 1]

# Ensure the directory for saving models exists
model_save_dir = "models_optimizer"

# Define the dataset and data loaders
train_dataset = TensorDataset(X_new_train, y_new_train)
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    worker_init_fn=seed_worker,
    generator=generator)

val_dataset = TensorDataset(X_val, y_val)
val_iter = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    worker_init_fn=seed_worker,
    generator=generator)

# Function to create the model
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0, std=0.01)  # Manual initialization of weights and bias for reproducability
        m.bias.data.fill_(0.0001)

def create_model(architecture):
    layers = []
    for i in range(len(architecture) - 1):
        layer = nn.Linear(architecture[i], architecture[i + 1])
        layers.append(layer)
        if i < len(architecture) - 2:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    model.apply(init_weights)   
    return model

# Loss function
criterion = nn.MSELoss()

# Gradient clipping value
max_grad_norm = 5.0

# Define optimizers with multiple learning rates
optimizers = [
    {"name": "SGD", "optimizer": SGD, "params": {"lr": lr}}
    for lr in [1e-3, 1e-2, 1e-1]
] + [
    {"name": "SGD_Momentum", "optimizer": SGD, "params": {"lr": lr, "momentum": 0.9}}
    for lr in [1e-3, 1e-2, 1e-1]
] + [
    {"name": "Adam", "optimizer": Adam, "params": {"lr": lr}}
    for lr in [1e-3, 1e-2, 1e-1]
]

# Schedulers
schedulers = [
    {"name": "None", "scheduler": None, "params": {}},
    {"name": "StepLR", "scheduler": StepLR, "params": {"step_size": 10, "gamma": 0.1}},
    {"name": "ReduceLROnPlateau", "scheduler": ReduceLROnPlateau, "params": {"factor": 0.5, "patience": 5}},
]

# Early stopping function
class EarlyStopping:
    def __init__(self, patience=40, delta=0.001, save_path="best_model.pth"):
        """
        Initializes early stopping.
        Args:
        - patience (int): How many epochs to wait before stopping when no improvement.
        - delta (float): Minimum change in the monitored metric to qualify as improvement.
        - save_path (str): Path to save the best model.
        """
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0  # Track the epoch of the best validation loss

    def __call__(self, val_loss, model, epoch):
        """
        Checks if training should stop.
        Args:
        - val_loss (float): Current validation loss.
        - model (nn.Module): The model to save when improvement occurs.
        - epoch (int): Current epoch number.
        """
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            print(f"Validation loss improved to {self.best_loss:.4f}. Saving model at epoch {epoch + 1}.")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered at epoch {epoch + 1}. Best validation loss was {self.best_loss:.4f} at epoch {self.best_epoch + 1}.")

# Training and validation function with best value tracking
def train_and_validate(
    model, train_iter, val_iter, num_epochs, optimizer_cls=None, optimizer_params=None, scheduler_cls=None, scheduler_params=None, early_stopping=None
):
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    scheduler = scheduler_cls(optimizer, **scheduler_params) if scheduler_cls else None

    train_losses, val_losses = [], []

    # Initialize best values
    best_val_loss = float("inf")
    best_train_loss = None
    best_epoch = 0
    best_model_state = None  # To store the state dict of the best model

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_iter:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_iter)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_iter:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if not torch.isfinite(loss):
                    print(f"NaN detected in validation loss at epoch {epoch + 1}. Skipping...")
                    loss = torch.tensor(float("inf"))  # Assign high loss value
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_iter)
        val_losses.append(avg_val_loss)

        # Update the best model and its state
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_train_loss = avg_train_loss
            best_epoch = epoch + 1
            best_model_state = model.state_dict()  # Save the best model state
            print(f"Validation loss improved to {best_val_loss:.4f} at epoch {epoch + 1}.")

        # Update scheduler if applicable
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        # Check early stopping
        if early_stopping:
            early_stopping(avg_val_loss, model, epoch)
            if early_stopping.early_stop:
                print(f"Early stopping triggered. Best validation loss: {best_val_loss:.4f}")
                break

        # Print intermediate results every 10 epochs
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses, best_train_loss, best_val_loss, best_epoch, best_model_state

# Run experiments for each optimizer and scheduler
results = []
for opt_config in optimizers:
    for sched_config in schedulers:
        print(f"=== Training with Optimizer: {opt_config['name']}, LR: {opt_config['params']['lr']}, Scheduler: {sched_config['name']} ===")
        model = create_model(architecture)
        
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=20, delta=0.001)
        
        train_losses, val_losses, best_train_loss, best_val_loss, best_epoch, best_model_state = train_and_validate(
            model,
            train_iter,
            val_iter,
            num_epochs,
            optimizer_cls=opt_config["optimizer"],
            optimizer_params=opt_config["params"],
            scheduler_cls=sched_config["scheduler"],
            scheduler_params=sched_config["params"],
            early_stopping=early_stopping
        )
        print(f"Best Train Loss: {best_train_loss:.4f}, Best Validation Loss: {best_val_loss:.4f} at Epoch {best_epoch}\n")
        
        # Save the best model state to a file
        model_save_path = f"{model_save_dir}/best_model_{opt_config['name']}_LR{opt_config['params']['lr']}_{sched_config['name']}.pth"
        print(f"Best model saved to {model_save_path}")

        # Save results
        results.append({
            "Optimizer": opt_config["name"],
            "Learning Rate": opt_config["params"]["lr"],
            "Scheduler": sched_config["name"],
            "Best Train Loss": best_train_loss,
            "Best Val Loss": best_val_loss,
            "Best Epoch": best_epoch,
            "Model Path": model_save_path
        })

# Store results in a DataFrame
results_df = pd.DataFrame(results)

# Display the final results
print("=== Optimizer and Scheduler Comparison Results ===")
print(results_df)

# %% [markdown]
# ## Task d)

# %%
set_seed(seed)
generator = torch.Generator()
generator.manual_seed(0)

# Retraining the best model to obtain the validation and train losses for the plot
train_dataset = TensorDataset(X_new_train, y_new_train)
train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    worker_init_fn=seed_worker,
    generator=generator)

val_dataset = TensorDataset(X_val, y_val)
val_iter = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    worker_init_fn=seed_worker,
    generator=generator)

batch_size = 50
num_epochs = 200
architecture = [8, 64, 32, 16, 1]

model_save_dir = "models_optimizer"

criterion = nn.MSELoss()
max_grad_norm = 5.0

optimizer_config = {"name": "SGD_Momentum", "optimizer": SGD, "params": {"lr": 1e-2, "momentum": 0.9}}
scheduler_config = {"name": "None", "scheduler": None, "params": {}}

results = []
train_losses = []
val_losses = []
best_train_loss = []
best_val_loss = []
best_epoch = []
best_model_state = []
        
model = create_model(architecture)
early_stopping = EarlyStopping(patience=40, delta=0.001)

train_losses, val_losses, best_train_loss, best_val_loss, best_epoch, best_model_state = train_and_validate(
    model,
    train_iter,
    val_iter,
    num_epochs,
    optimizer_cls=optimizer_config["optimizer"],
    optimizer_params=optimizer_config["params"],
    scheduler_cls=scheduler_config["scheduler"],
    scheduler_params=scheduler_config["params"],
    early_stopping=early_stopping
)

# Plot Training and Validation Losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Training and Validation Losses")
plt.legend()
plt.grid()
plt.show()

# %%
set_seed(seed)
generator = torch.Generator()
generator.manual_seed(0)

# Combine training and validation data
X_combined = torch.cat([X_new_train, X_val], dim=0)
y_combined = torch.cat([y_new_train, y_val], dim=0)

combined_dataset = TensorDataset(X_combined, y_combined)
combined_iter = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    worker_init_fn=seed_worker,
    generator=generator)

# Initialize and train the final model
batch_size = 50
num_epochs = 200
architecture = [8, 64, 32, 16, 1]
max_grad_norm = 5.0

final_model = create_model(architecture)
optimizer_config = {"name": "SGD_Momentum", "optimizer": SGD, "params": {"lr": 1e-2, "momentum": 0.9}}
final_optimizer = optimizer_config["optimizer"](final_model.parameters(), **optimizer_config["params"])
final_criterion = nn.MSELoss()

schedulers = []
scheduler = None

final_train_losses = []
print("Final model training started.")
for epoch in range(num_epochs):
    final_model.train()
    epoch_loss = 0
    for X_batch, y_batch in combined_iter:
        final_optimizer.zero_grad()
        y_pred = final_model(X_batch)
        loss = final_criterion(y_pred, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(final_model.parameters(), max_grad_norm)
        final_optimizer.step()
        epoch_loss += loss.item()

    final_train_losses.append(epoch_loss / len(combined_iter))

    if scheduler:
        scheduler.step(epoch_loss / len(combined_iter))
    
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(combined_iter):.4f}")

print("Final model training completed.")

# Test the model
final_model.eval()
y_test_pred = None
test_loss = None
with torch.no_grad():
    y_test_pred = final_model(X_test)
    test_loss = final_criterion(y_test_pred, y_test).item()

print(f"Final Test Loss: {test_loss:.4f}")


# %%
# Scatter plot: Predictions vs Ground Truth
y_test_pred = y_test_pred.flatten()
y_test_true = y_test.flatten()

plt.figure(figsize=(8, 8))
plt.scatter(y_test_pred, y_test_true, alpha=0.6, edgecolors='k', label="Predictions")
plt.plot([y_test_true.min(), y_test_true.max()], [y_test_true.min(), y_test_true.max()], color='red', linestyle='--', label="Ideal")
plt.xlabel("Model Predictions")
plt.ylabel("Ground Truth")
plt.title("Predictions vs Ground Truth (Test Set)")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## Task f)

# %%
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

set_seed(seed)
generator = torch.Generator()
generator.manual_seed(seed)


# Step 1: Load and Preprocess the Dataset
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=302)

# Sanity check: Dataset shapes
print(f"Initial shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Sanity check: Feature scaling
print(f"After scaling - X_train mean: {X_train.mean():.2f}, std: {X_train.std():.2f}")
print(f"After scaling - X_test mean: {X_test.mean():.2f}, std: {X_test.std():.2f}")

# Transform target into binary labels
y_train = (y_train >= 2).astype(int)  # 2 corresponds to $200,000
y_test = (y_test >= 2).astype(int)

# Sanity check: Unique values in target variables
print(f"Unique values in y_train: {set(y_train)}, Unique values in y_test: {set(y_test)}")

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Print dimensions for verification
print(f"Final shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

# Step 2: Define the Model Architecture
class BinaryClassifier(nn.Module):
    def __init__(self, architecture):
        super(BinaryClassifier, self).__init__()
        layers = []
        for i in range(len(architecture) - 1):
            layers.append(nn.Linear(architecture[i], architecture[i + 1]))
            if i < len(architecture) - 2:
                layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

architecture = [8, 64, 32, 16, 1]  # Same as regression, with 1 output for binary classification
model = BinaryClassifier(architecture)
print("Model architecture:")
print(model)

# Step 3: Define Loss, Optimizer, DataLoader, and Scheduler
criterion = nn.BCEWithLogitsLoss()
optimizer_config = {"name": "SGD_Momentum", "optimizer": SGD, "params": {"lr": 1e-2, "momentum": 0.9}}
optimizer = optimizer_config["optimizer"](model.parameters(), **optimizer_config["params"])

scheduler = None
batch_size = 50

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X_test, y_test)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Step 4: Train the Model with Early Stopping
num_epochs = 200
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience = 8
counter = 0
best_model_path = 'best_model.pth'

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)

    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            epoch_val_loss += loss.item()
    epoch_val_loss /= len(val_loader)
    val_losses.append(epoch_val_loss)

    #scheduler.step(epoch_val_loss)

    if epoch % 5 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

# Load the best model
model.load_state_dict(torch.load(best_model_path))
# Step 5: Evaluate the Model
model.eval()
y_test_pred = []
y_test_true = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_pred = torch.sigmoid(model(X_batch))
        y_test_pred.extend(y_pred.squeeze().tolist())
        y_test_true.extend(y_batch.squeeze().tolist())

# Convert predictions to binary labels
y_test_pred = np.array(y_test_pred)
y_test_true = np.array(y_test_true)
y_test_pred_labels = (y_test_pred >= 0.5).astype(int)

# Calculate metrics
accuracy = accuracy_score(y_test_true, y_test_pred_labels)
precision = precision_score(y_test_true, y_test_pred_labels, zero_division=0)
recall = recall_score(y_test_true, y_test_pred_labels, zero_division=0)
f1 = f1_score(y_test_true, y_test_pred_labels, zero_division=0)
roc_auc = roc_auc_score(y_test_true, y_test_pred)

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Generate and plot confusion matrix
cm = confusion_matrix(y_test_true, y_test_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Below $200k', 'Above $200k'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Step 6: Plot Training and Validation Loss
plt.figure()
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.grid()
plt.show()

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test_true, y_test_pred)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# %%



