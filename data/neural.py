"""
Neural Data Container (Stub)

Extensible container for trial-aligned neural recordings.
Designed to accommodate calcium imaging (dF/F) and
electrophysiology (spike times, firing rates) when ready.

Structure: neuron × trial × within_trial_timepoints × values
with epoch labels (stimulus, delay, choice, ITI).

Not yet implemented — this defines the interface and extension points.
Fill in when you have real neural data to work with.

Usage (future):
    from behav_utils.data.neural import NeuralData

    neural = NeuralData.from_suite2p(suite2p_dir, session)
    neural.n_neurons      # → 200
    neural.get_epoch('stimulus')  # → (n_neurons, n_trials, n_timepoints)
    neural.get_neuron(42)         # → (n_trials, n_timepoints)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple


@dataclass
class Epoch:
    """
    A named temporal epoch within a trial.

    Attributes:
        name: Epoch label (e.g., 'stimulus', 'delay', 'choice', 'iti')
        start_idx: First timepoint index within trial
        end_idx: Last timepoint index (exclusive)
        duration_s: Duration in seconds (for reference)
    """
    name: str
    start_idx: int
    end_idx: int
    duration_s: Optional[float] = None

    @property
    def n_timepoints(self) -> int:
        return self.end_idx - self.start_idx


@dataclass
class NeuralData:
    """
    Trial-aligned neural recordings for one session.

    Core data:
        traces: np.ndarray of shape (n_neurons, n_trials, n_timepoints)
                Values are dF/F for imaging, firing rate for ephys,
                or raw voltage — interpretation depends on data_type.

    Metadata:
        neuron_ids: Identifiers for cross-session tracking
        neuron_types: 'excitatory', 'inhibitory', 'unclassified'
        epochs: Named temporal epochs within each trial
        sampling_rate: Hz
        data_type: 'calcium_imaging', 'ephys_rate', 'ephys_spikes'

    Linked to behaviour via trial indices — NeuralData.trial_indices[i]
    corresponds to TrialData index for neural trial i.
    """
    # Core data
    traces: np.ndarray  # (n_neurons, n_trials, n_timepoints)

    # Trial alignment
    trial_indices: np.ndarray  # maps neural trials → TrialData indices

    # Neuron metadata
    neuron_ids: np.ndarray = field(default_factory=lambda: np.array([]))
    neuron_types: np.ndarray = field(default_factory=lambda: np.array([]))
    # 'excitatory', 'inhibitory', 'unclassified'

    # Temporal structure
    epochs: List[Epoch] = field(default_factory=list)
    sampling_rate: float = 30.0  # Hz
    timepoints_per_trial: int = 0

    # Recording metadata
    data_type: str = 'calcium_imaging'
    # 'calcium_imaging': dF/F values
    # 'ephys_rate': firing rates (Hz)
    # 'ephys_spikes': spike counts per bin

    session_id: Optional[str] = None

    def __post_init__(self):
        if self.traces.ndim != 3:
            raise ValueError(
                f"traces must be 3D (n_neurons, n_trials, n_timepoints), "
                f"got shape {self.traces.shape}"
            )
        if self.timepoints_per_trial == 0:
            self.timepoints_per_trial = self.traces.shape[2]
        if len(self.neuron_ids) == 0:
            self.neuron_ids = np.arange(self.n_neurons)
        if len(self.neuron_types) == 0:
            self.neuron_types = np.full(self.n_neurons, 'unclassified',
                                        dtype=object)

    @property
    def n_neurons(self) -> int:
        return self.traces.shape[0]

    @property
    def n_trials(self) -> int:
        return self.traces.shape[1]

    @property
    def n_timepoints(self) -> int:
        return self.traces.shape[2]

    @property
    def n_excitatory(self) -> int:
        return int(np.sum(self.neuron_types == 'excitatory'))

    @property
    def n_inhibitory(self) -> int:
        return int(np.sum(self.neuron_types == 'inhibitory'))

    # ── Access ──────────────────────────────────────────────────────────────

    def get_neuron(self, neuron_idx: int) -> np.ndarray:
        """Get traces for one neuron: (n_trials, n_timepoints)."""
        return self.traces[neuron_idx]

    def get_trial(self, trial_idx: int) -> np.ndarray:
        """Get traces for one trial: (n_neurons, n_timepoints)."""
        return self.traces[:, trial_idx, :]

    def get_epoch(self, epoch_name: str) -> np.ndarray:
        """
        Get traces for a named epoch: (n_neurons, n_trials, epoch_timepoints).
        """
        for epoch in self.epochs:
            if epoch.name == epoch_name:
                return self.traces[:, :, epoch.start_idx:epoch.end_idx]
        raise ValueError(
            f"Epoch '{epoch_name}' not found. "
            f"Available: {[e.name for e in self.epochs]}"
        )

    def get_neurons_by_type(self, neuron_type: str) -> np.ndarray:
        """
        Get traces for neurons of a specific type.
        Returns: (n_matched, n_trials, n_timepoints)
        """
        mask = self.neuron_types == neuron_type
        return self.traces[mask]

    # ── Factory methods (to be implemented per data source) ─────────────────

    @classmethod
    def from_suite2p(cls, suite2p_dir, trial_indices, epochs=None,
                     **kwargs) -> 'NeuralData':
        """Load from Suite2P output. Not yet implemented."""
        raise NotImplementedError(
            "Suite2P loading not yet implemented. "
            "See behav_utils.data.neural for the interface."
        )

    @classmethod
    def from_caiman(cls, caiman_output, trial_indices, epochs=None,
                    **kwargs) -> 'NeuralData':
        """Load from CaImAn output. Not yet implemented."""
        raise NotImplementedError(
            "CaImAn loading not yet implemented. "
            "See behav_utils.data.neural for the interface."
        )

    @classmethod
    def from_arrays(
        cls,
        traces: np.ndarray,
        trial_indices: np.ndarray,
        neuron_types: Optional[np.ndarray] = None,
        epochs: Optional[List[Epoch]] = None,
        **kwargs,
    ) -> 'NeuralData':
        """
        Create from raw arrays. The general-purpose constructor.

        Args:
            traces: (n_neurons, n_trials, n_timepoints)
            trial_indices: maps neural trials to behaviour trial indices
            neuron_types: 'excitatory'/'inhibitory'/'unclassified' per neuron
            epochs: list of Epoch objects defining temporal structure
        """
        return cls(
            traces=traces,
            trial_indices=trial_indices,
            neuron_types=neuron_types if neuron_types is not None else np.array([]),
            epochs=epochs if epochs is not None else [],
            **kwargs,
        )
