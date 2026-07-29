from typing import Callable, NotRequired, TypedDict, Union
import pandas as pd


class FeatureMap(TypedDict):
    name: str
    features: NotRequired[list[str]]
    how: Union[str, Callable[[pd.DataFrame], pd.Series]]
