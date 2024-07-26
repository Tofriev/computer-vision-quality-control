# %%
import sys

sys.path.append("../")
from src.data_preprocessing.dataset import MyDataset
from src.pipeline import Pipeline

# %%
pipe = Pipeline()
pipe.run()
