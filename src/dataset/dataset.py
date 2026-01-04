import pandas as pd
from io import StringIO
from torch.utils.data import Dataset

class QuestionDataset(Dataset):
    def __init__(self, df:pd.DataFrame) -> None:
        self._df = df
    
    def __len__(self):
        return len(self._df)
    
    def __getitem__(self, key):
        return self._df[key]
    
    def __str__(self) -> str:
        output = StringIO()
        self._df.info(buf=output)
        return output.getvalue()
    
    @classmethod
    def load(cls, path:str, dataset_type:str):
        df = pd.read_csv(path)
        if dataset_type.lower() == "advbench":
            df.columns = ["input", "target"]
            return cls(df.iloc[:, 0:1])
        elif dataset_type.lower() == "fr":
            return cls(df)
        elif dataset_type.lower() == "xstest":
            df.rename(columns={"prompt":"input"})
            return cls(df)
        elif dataset_type.lower() == "orbench":
            df.rename(columns={"prompt":"input"})
            return cls(df)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def save(self, path:str):
        self._df.to_csv(path, index=False)
