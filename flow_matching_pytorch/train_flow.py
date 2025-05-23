import sys
sys.path.append('..')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher, TargetConditionalFlowMatcher

from grids import SpectralDataset


class FlowModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.input_dim = input_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Condition embedding
        self.cond_embed = nn.Sequential(
            nn.Linear(2, hidden_dim),  # 2 for age and metallicity
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Main network
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim + 2*hidden_dim, hidden_dim))  # input + time_embed + cond_embed
        layers.append(nn.SiLU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, t, x, y):
        # t: time (batch_size,)
        # x: current state (batch_size, input_dim)
        # y: condition (batch_size, 2) - age and metallicity
        
        # Reshape time to (batch_size, 1)
        t = t.view(-1, 1)
        
        # Embed time
        t_embed = self.time_embed(t)
        
        # Embed condition
        cond_embed = self.cond_embed(y)
        
        # Concatenate all inputs
        h = torch.cat([x, t_embed, cond_embed], dim=1)
        
        # Pass through network
        return self.net(h)

def create_flow_model(device, input_dim, hidden_dim=128, num_layers=4):
    return FlowModel(input_dim, hidden_dim, num_layers).to(device)

def main():
    # Hyperparameters
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 60
    patience = 10  # Early stopping patience
    min_delta = 1e-4  # Minimum change in loss to be considered an improvement
    
    # Learning rate scheduler parameters
    lr_patience = 3  # Number of epochs to wait before reducing learning rate
    lr_factor = 0.5  # Factor to reduce learning rate by
    min_lr = 1e-7  # Minimum learning rate

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Create dataset and dataloader
    grid_dir = '../synthesizer_grids/grids/'
    dataset = SpectralDataset(f'{grid_dir}/bc03-2016-Miles_chabrier-0.1,100.hdf5')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create flow model
    flow_model = create_flow_model(device, input_dim=dataset.n_wavelength)
    # cfm = ConditionalFlowMatcher(sigma=0.1)
    cfm = TargetConditionalFlowMatcher(sigma=0.1)
    # node = NeuralODE(flow_model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

    # Training setup
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=lr_factor, patience=lr_patience, 
        min_lr=min_lr,
    )
    
    # Early stopping variables
    best_loss = float('inf')
    best_epoch = 0
    no_improve_epochs = 0
    
    # Training loop
    for epoch in range(num_epochs):
        flow_model.train()
        total_loss = 0
        for i, (conditions, spectra) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Move data to device
            conditions = conditions.to(device)  # age and metallicity
            spectra = spectra.to(device)  # target spectra
            
            # Generate random initial state
            x0 = torch.randn_like(spectra)
            
            # Sample points along the flow
            t, xt, ut = cfm.sample_location_and_conditional_flow(x0, spectra)
            
            # Compute model prediction
            vt = flow_model(t, xt, conditions)
            
            # Compute loss
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            print(f"epoch: {epoch}, steps: {i}, loss: {loss.item():.4f}", end="\r")
        
        avg_loss = total_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}, Learning Rate: {current_lr:.2e}")
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        # Early stopping check
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            best_epoch = epoch
            no_improve_epochs = 0
            # Save best model
            torch.save(flow_model.state_dict(), 'models/spectral_flow_model_best.pt')
        else:
            no_improve_epochs += 1
            
        if no_improve_epochs >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"Best loss: {best_loss:.4f} at epoch {best_epoch + 1}")
            break
    
    # Load best model
    flow_model.load_state_dict(torch.load('models/spectral_flow_model_best.pt'))
    # Save final model
    torch.save(flow_model.state_dict(), 'models/spectral_flow_model.pt')

if __name__ == "__main__":
    main() 