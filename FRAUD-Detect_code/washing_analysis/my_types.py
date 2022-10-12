from typing import Protocol, List, Union, Type
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from fairlearn.reductions import ExponentiatedGradient, GridSearch


class Mitigator_Proba(Protocol):
    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        raise NotImplementedError

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike, sensitive_features: npt.ArrayLike):
        raise NotImplementedError

    def predict_probas(self, X: npt.ArrayLike) -> List[tuple]:
        raise NotImplementedError


class Mitigator_Pmf_Proba(Protocol):
    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        raise NotImplementedError

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike, sensitive_features: npt.ArrayLike):
        raise NotImplementedError

    def _pmf_predict(self, X: npt.ArrayLike) -> List[tuple]:
        raise NotImplementedError


Mitigator = Union[Mitigator_Proba, Mitigator_Pmf_Proba]
Mit_type = Type[Union[ExponentiatedGradient, GridSearch]]
Explainer = Union[LogisticRegression, DecisionTreeClassifier]