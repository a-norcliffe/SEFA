"""Models dict so that we can use the model names as strings in arguments
to easily select a model from the command line.
"""


from models.acflow import ACFlow
from models.dime import DIME
from models.eddi import EDDI
from models.feature_space_ablation import FeatureSpaceAblation
from models.fixed_mlp import FixedMLP
from models.gdfs import GDFS
from models.gsmrl import GSMRL
from models.opportunistic_rl import OpportunisticRL
from models.random_ordering import RandomOrdering
from models.sefa import SEFA
from models.vae import VAE



models_dict = {
  "acflow": ACFlow,
  "dime": DIME,
  "eddi": EDDI,
  "feature_space_ablation": FeatureSpaceAblation,
  "fixed_mlp": FixedMLP,
  "gdfs": GDFS,
  "gsmrl": GSMRL,
  "opportunistic": OpportunisticRL,
  "random": RandomOrdering,
  "sefa": SEFA,
  "vae": VAE,
}