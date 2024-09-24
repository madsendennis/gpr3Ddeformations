from trimesh import Trimesh
from vectorization import vectorize, unvectorize
import numpy as np
import h5py


class GPMM:
    def __init__(
        self,
        reference_points: np.ndarray,
        reference_cells: np.ndarray,
        model_mean_deformation: np.ndarray,
        model_basis: np.ndarray,
        model_variance: np.ndarray,
    ):
        self.reference_points = reference_points
        self.reference_cells = reference_cells
        self.model_mean_deformation = model_mean_deformation
        self.model_basis = model_basis
        self.model_variance = model_variance
        self.rank: int = len(self.model_variance)
        self.dim: int = len(self.reference_points[0])

    def mean(self) -> Trimesh:
        c = np.zeros(self.rank)
        return self.instance(c)

    # def cov(self, ptId1: int, ptId2: int) -> np.array:
    #     pass

    def sample(self, scaling: float = 1.0) -> Trimesh:
        c = np.random.normal(0, 1, self.rank) * scaling
        return self.instance(c)

    def instance(self, c: np.array) -> Trimesh:
        scaled_c = c * np.sqrt(self.model_variance)
        deformation = self.model_mean_deformation + self.model_basis @ scaled_c
        return self._to_trimesh(deformation)

    # def pdf(self, coefficients: np.array) -> float:
    #     pass

    # def truncate(self, new_rank: int) -> GPMM:
    #     pass

    # def posterior(self) -> GPMM:
    #     pass

    # def marginal(self) -> GPMM:
    #     pass

    # def change_reference(self) -> GPMM:
    #     pass

    # def decimate(self, target_num_vertices: int) -> GPMM:
    #     # update gpmm with new decimated reference
    #     pass

    def _to_trimesh(self, deformation):
        points = self.reference_points + unvectorize(deformation, self.dim)
        return Trimesh(vertices=points, faces=self.reference_cells)


def gpmm_from_h5(file: str) -> GPMM:
    with h5py.File(file, "r") as file:
        reference_points = np.array(file["representer"]["points"]).T
        reference_points_vec = vectorize(reference_points)
        reference_cells = np.array(file["representer"]["cells"]).T
        model_mean = np.array(file["model"]["mean"])
        model_basis = np.array(file["model"]["pcaBasis"])
        model_variance = np.array(file["model"]["pcaVariance"])
    model_mean_deformation = model_mean - reference_points_vec

    return GPMM(
        reference_points,
        reference_cells,
        model_mean_deformation,
        model_basis,
        model_variance,
    )
