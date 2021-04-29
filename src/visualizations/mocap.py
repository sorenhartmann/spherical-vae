import torch
from pathlib import Path
from src.data.mocap import MotionCaptureDataset
from src.experiments.run_walk import model_args
import seaborn as sns
from src.models.vae import VariationalAutoencoder

run_dir = Path(__file__).parents[2] / "runs" / "run-walk"

if __name__ == "__main__":
    
    run_data = MotionCaptureDataset("09", test=True)
    walk_data = MotionCaptureDataset("08", test=True)

    vae = VariationalAutoencoder(
        latent_dim=2, **model_args
    )

    vae_state_dict = torch.load( run_dir / "best_vae.pt", map_location=torch.device('cpu')  )
    vae.load_state_dict( vae_state_dict )

    sns.scatterplot(*vae(run_data.X)["z"].detach().numpy().T)
    sns.scatterplot(*vae(walk_data.X)["z"].detach().numpy().T)
