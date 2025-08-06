#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Piece-wise random fields class."""

from copy import deepcopy

from queens.parameters.random_fields._random_field import RandomField


class PieceWise(RandomField):
    """Piece-wise random field class.

    The field is constructed at each node or element which is given by the
    coordinates. Each piece is assumed independently. Equals creating
    random variables for each point in coords.

    Attributes:
            distribution (obj): Dummy distribution with correct dimension
            latent_1d_distribution: (obj): QUEENS distribution object of latent space variables
    """

    def __init__(self, coords, latent_1d_distribution):
        """Initialize RF object.

        Args:
            coords (dict): Dictionary with coordinates of discretized random field and the
                           corresponding keys
            latent_1d_distribution (Distribution): Latent 1d distribution that is used for all
                                                   variables.
        """
        super().__init__(coords)
        self.dimension = self.dim_coords
        self.latent_1d_distribution = latent_1d_distribution
        self.distribution = deepcopy(latent_1d_distribution)
        self.distribution.dimension = self.dimension

    def draw(self, num_samples):
        """Draw samples from the latent representation of the random field.

        Args:
            num_samples: Number of draws of latent random samples
        Returns:
            samples (np.ndarray): Drawn samples
        """
        samples = self.latent_1d_distribution.draw(num_samples * self.dimension).reshape(
            num_samples, self.dimension
        )
        return samples

    def logpdf(self, samples):
        """Get joint logpdf of latent space."""
        return (
            self.latent_1d_distribution.logpdf(samples.reshape(-1, 1))
            .reshape(samples.shape)
            .sum(axis=1)
        )

    def grad_logpdf(self, samples):
        """Get gradient of joint logpdf of latent space."""
        return self.latent_1d_distribution.grad_logpdf(samples.reshape(-1, 1)).reshape(
            samples.shape
        )

    def expanded_representation(self, samples):
        """Expand latent representation of sample.

        Args:
            samples (np.ndarray): latent representation of sample

        Returns:
            samples (np.ndarray): Expanded representation of sample
        """
        return samples

    def latent_gradient(self, upstream_gradient):
        """Gradient of the field with respect to the latent parameters.

        Args:
            upstream_gradient (np.ndarray): Gradient with respect to all coords of the field

        Returns:
            upstream_gradient (np.ndarray): Graident of the field with respect to the latent
            parameters
        """
        return upstream_gradient
