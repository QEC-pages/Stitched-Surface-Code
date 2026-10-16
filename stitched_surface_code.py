# Copyright 2026 Google LLC (or relevant authors)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Memory circuits of the stitched surface code: four rotated surface-code
patches of distance t concatenated with the [[4,2,2]] code, whose two
weight-4t generators are measured with verified GHZ states on corridors of
ancillary qubits between the patches (M. A. Shalby et al., "Stitched surface
codes: planar concatenation through GHZ corridors", 2026).

build_circuit(t, rv, ro, memory) returns a noiseless Stim circuit with all
detectors and both logical observables.  Data qubits are initialized by
sweep-controlled X gates; no qubit is reset before the first
syndrome-extraction cycle, so ancillas start in Stim's implicit |0> state.
Detectors are declared on named measurement records at the point in the
circuit where they become deterministic.  The construction uses only fixed
tables and closed-form coordinate formulas; it depends only on Stim.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from numbers import Integral
from typing import Iterable, Literal, Sequence

import stim

Coordinate = tuple[int, int]
Pair = tuple[int, int]
Basis = Literal["X", "Z"]
Memory = Literal["ZZ", "XX"]


def _is_x_memory(memory: Memory) -> bool:
    """Whether preparation, readout, and logical observables use X basis."""
    return memory == "XX"


def _uses_zz_flow(memory: Memory) -> bool:
    """Whether the physical corridor schedule follows the ZZ ordering."""
    return memory == "ZZ"


# =============================================================================
# Physical layout and coordinates
# =============================================================================


PATCH_IDS = (1, 2, 3, 4)
PATCH_OFFSETS = {
    1: (0, 0),
    2: (1, 0),
    3: (0, 1),
    4: (1, 1),
}


@dataclass(frozen=True)
class _Patch:
    data: tuple[Coordinate, ...]
    x_ancillas: tuple[Coordinate, ...]
    z_ancillas: tuple[Coordinate, ...]

    @cached_property
    def ancillas(self) -> tuple[Coordinate, ...]:
        return tuple(sorted((*self.x_ancillas, *self.z_ancillas)))


@dataclass(frozen=True)
class _Corridor:
    first_flanks: tuple[int, ...]
    second_flanks: tuple[int, ...]
    centers: tuple[int, ...]
    first_wing: int
    second_wing: int

    @cached_property
    def flanks(self) -> tuple[int, ...]:
        return tuple(sorted((*self.first_flanks, *self.second_flanks)))

    @cached_property
    def flags(self) -> tuple[int, int]:
        return tuple(sorted((self.first_wing, self.second_wing)))

    @cached_property
    def resource(self) -> tuple[int, ...]:
        return tuple(sorted((*self.flanks, *self.centers, *self.flags)))


@dataclass(frozen=True)
class _Geometry:
    d: int
    coordinate_to_qubit: dict[Coordinate, int]
    qubit_to_coordinate: dict[int, Coordinate]
    patches: dict[int, _Patch]
    vertical: _Corridor
    horizontal: _Corridor
    vertical_disabled: frozenset[int]
    horizontal_disabled: frozenset[int]
    ancilla_basis: dict[int, Basis]
    ancilla_data_neighbors: dict[int, tuple[int, ...]]

    def q(self, coordinate: Coordinate) -> int:
        return self.coordinate_to_qubit[coordinate]

    @cached_property
    def data(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                self.q(coordinate) for patch in self.patches.values() for coordinate in patch.data
            )
        )

    @cached_property
    def ancillas(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    self.q(coordinate)
                    for patch in self.patches.values()
                    for coordinate in patch.ancillas
                }
            )
        )


def _patch(d: int, x_offset: int, y_offset: int) -> _Patch:
    data = tuple(
        sorted((2 * i + 1 + x_offset, 2 * j + 1 + y_offset) for i in range(d) for j in range(d))
    )
    x_ancillas: list[Coordinate] = []
    z_ancillas: list[Coordinate] = []
    for i in range(d + 1):
        for j in range(d + 1):
            on_left_or_right = i in (0, d)
            on_top_or_bottom = j in (0, d)
            is_x = (i % 2) != (j % 2)
            if (on_left_or_right and is_x) or (on_top_or_bottom and not is_x):
                continue
            target = x_ancillas if is_x else z_ancillas
            target.append((2 * i + x_offset, 2 * j + y_offset))
    return _Patch(data, tuple(sorted(x_ancillas)), tuple(sorted(z_ancillas)))


def _build_geometry(d: int) -> _Geometry:
    """Build checkerboard-oriented patches around fixed V/H corridors.

    Diagonally opposite P2/P3 have rotated, Pauli-dual boundaries. For odd d
    this is the identical patch; for even d it balances the corridor-facing
    checks. Qubit IDs are allocated on the original scaffold, then only
    vacated exterior syndrome sites move. Data and corridor IDs stay fixed.
    """
    if d < 2:
        raise ValueError("d must be at least 2")
    separation = 2 * d + 2
    patches = {
        patch_id: _patch(d, gx * separation, gy * separation)
        for patch_id, (gx, gy) in PATCH_OFFSETS.items()
    }

    coordinate_to_qubit: dict[Coordinate, int] = {}
    qubit_to_coordinate: dict[int, Coordinate] = {}

    def add(coordinates: Iterable[Coordinate]) -> dict[Coordinate, int]:
        result: dict[Coordinate, int] = {}
        for coordinate in sorted(coordinates, key=lambda value: (value[1], value[0])):
            if coordinate not in coordinate_to_qubit:
                qubit = len(coordinate_to_qubit)
                coordinate_to_qubit[coordinate] = qubit
                qubit_to_coordinate[qubit] = coordinate
            result[coordinate] = coordinate_to_qubit[coordinate]
        return result

    for patch_id in PATCH_IDS:
        patch = patches[patch_id]
        add((*patch.data, *patch.ancillas))

    n = 2 * d
    center = 2 * d + 1
    vertical_left = tuple((2 * d, 2 * (j + 1)) for j in range(n))
    vertical_right = tuple((2 * d + 2, 2 * (j + 1)) for j in range(n))
    vertical_centers = tuple((center, 2 * j + 1) for j in range(n + 1))
    vertical_wings = ((center - 2, center), (center + 2, center))
    horizontal_top = tuple((2 * (j + 1), 2 * d) for j in range(n))
    horizontal_bottom = tuple((2 * (j + 1), 2 * d + 2) for j in range(n))
    horizontal_centers = tuple((2 * j + 1, center) for j in range(n + 1))
    horizontal_wings = ((center, center - 2), (center, center + 2))

    vertical_left_map = add(vertical_left)
    vertical_right_map = add(vertical_right)
    vertical_center_map = add(vertical_centers)
    vertical_wing_map = add(vertical_wings)
    horizontal_top_map = add(horizontal_top)
    horizontal_bottom_map = add(horizontal_bottom)
    horizontal_center_map = add(horizontal_centers)
    horizontal_wing_map = add(horizontal_wings)
    vertical = _Corridor(
        tuple(vertical_left_map[q] for q in vertical_left),
        tuple(vertical_right_map[q] for q in vertical_right),
        tuple(vertical_center_map[q] for q in vertical_centers),
        vertical_wing_map[vertical_wings[0]],
        vertical_wing_map[vertical_wings[1]],
    )
    horizontal = _Corridor(
        tuple(horizontal_top_map[q] for q in horizontal_top),
        tuple(horizontal_bottom_map[q] for q in horizontal_bottom),
        tuple(horizontal_center_map[q] for q in horizontal_centers),
        horizontal_wing_map[horizontal_wings[0]],
        horizontal_wing_map[horizontal_wings[1]],
    )

    for patch_id in (2, 3):
        gx, gy = PATCH_OFFSETS[patch_id]
        ox, oy = gx * separation, gy * separation

        def rotate(xy: Coordinate) -> Coordinate:
            return ox + 2 * d - (xy[1] - oy), oy + xy[0] - ox

        old = patches[patch_id]
        patches[patch_id] = _Patch(
            old.data,
            tuple(sorted(map(rotate, old.z_ancillas))),
            tuple(sorted(map(rotate, old.x_ancillas))),
        )
    needed = {xy for patch in patches.values() for xy in (*patch.data, *patch.ancillas)} | {
        qubit_to_coordinate[q] for q in (*vertical.resource, *horizontal.resource)
    }
    removed = sorted(set(coordinate_to_qubit) - needed)
    added = sorted(needed - set(coordinate_to_qubit))
    if len(removed) != len(added):
        raise AssertionError("balanced layout changed the qubit count")
    for old, new in zip(removed, added):
        qubit = coordinate_to_qubit.pop(old)
        coordinate_to_qubit[new] = qubit
        qubit_to_coordinate[qubit] = new

    vertical_disabled_coordinates: set[Coordinate] = set()
    horizontal_disabled_coordinates: set[Coordinate] = set()
    for patch_id, patch in patches.items():
        vertical_boundary = 2 * d if patch_id in (1, 3) else separation
        horizontal_boundary = 2 * d if patch_id in (1, 2) else separation
        vertical_disabled_coordinates |= {q for q in patch.ancillas if q[0] == vertical_boundary}
        horizontal_disabled_coordinates |= {
            q for q in patch.ancillas if q[1] == horizontal_boundary
        }
    if not len(vertical_disabled_coordinates) == len(horizontal_disabled_coordinates) == 2 * d - 2:
        raise AssertionError("V/H corridor-facing boundary counts are not balanced")

    ancilla_basis: dict[int, Basis] = {}
    ancilla_data_neighbors: dict[int, tuple[int, ...]] = {}
    for patch in patches.values():
        data_coordinates = set(patch.data)
        for basis, ancillas in (("X", patch.x_ancillas), ("Z", patch.z_ancillas)):
            for coordinate in ancillas:
                ancilla = coordinate_to_qubit[coordinate]
                ancilla_basis[ancilla] = basis
                ancilla_data_neighbors[ancilla] = tuple(
                    coordinate_to_qubit[(coordinate[0] + dx, coordinate[1] + dy)]
                    for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1))
                    if (coordinate[0] + dx, coordinate[1] + dy) in data_coordinates
                )

    return _Geometry(
        d=d,
        coordinate_to_qubit=coordinate_to_qubit,
        qubit_to_coordinate=qubit_to_coordinate,
        patches=patches,
        vertical=vertical,
        horizontal=horizontal,
        vertical_disabled=frozenset(
            map(coordinate_to_qubit.__getitem__, vertical_disabled_coordinates)
        ),
        horizontal_disabled=frozenset(
            map(coordinate_to_qubit.__getitem__, horizontal_disabled_coordinates)
        ),
        ancilla_basis=ancilla_basis,
        ancilla_data_neighbors=ancilla_data_neighbors,
    )


# =============================================================================
# Fixed module schedules: patch flow and V/H corridor tables
# =============================================================================


PLUS_PATCH_ORDERS = {
    1: (
        ((+1, +1), (-1, +1), (+1, -1), (-1, -1)),
        ((+1, +1), (+1, -1), (-1, +1), (-1, -1)),
    ),
    6: (
        ((-1, -1), (+1, -1), (-1, +1), (+1, +1)),
        ((-1, -1), (-1, +1), (+1, -1), (+1, +1)),
    ),
}
MINUS_PATCH_ORDERS = {
    1: (
        ((-1, +1), (+1, +1), (-1, -1), (+1, -1)),
        ((-1, +1), (-1, -1), (+1, +1), (+1, -1)),
    ),
    6: (
        ((+1, -1), (-1, -1), (+1, +1), (-1, +1)),
        ((+1, -1), (+1, +1), (-1, -1), (-1, +1)),
    ),
}

# P1,P2,P3,P4.  Each initialization extraction uses the same patch partition
# as the first corridor module it feeds: ZZ_PREP enters V and XX_PREP enters
# H.  V groups the patches on each side of its corridor; H does the same for
# its corridor.  No physical patch is singled out by any module.
PATCH_POLICY = {
    "ZZ_PREP": (1, 6, 1, 6),
    "XX_PREP": (1, 1, 6, 6),
    "V": (1, 6, 1, 6),
    "H": (1, 1, 6, 6),
}

# Even-distance checkerboard patches need the lower-left sweep reflected to
# leave a legal terminal coupling slot. One policy serves every even d;
# there is no distance-specific generator or runtime schedule search.
EVEN_V_PATCH_ORDERS = (
    PLUS_PATCH_ORDERS[1],
    PLUS_PATCH_ORDERS[6],
    MINUS_PATCH_ORDERS[6],
    PLUS_PATCH_ORDERS[6],
)


def _dual_layers(
    geometry: _Geometry, layers: tuple[tuple[Pair, ...], ...]
) -> tuple[tuple[Pair, ...], ...]:
    """Rotate the same fixed device by screen-r270 and exchange X/Z roles."""
    extent = 4 * geometry.d + 2
    image = {q: geometry.q((y, extent - x)) for q, (x, y) in geometry.qubit_to_coordinate.items()}
    return tuple(tuple((image[t], image[c]) for c, t in layer) for layer in layers)


def _vertical_prep_layers(geometry: _Geometry) -> tuple[tuple[Pair, ...], ...]:
    """Four-layer centered-echo V-cat fusion tree."""
    left = geometry.vertical.first_flanks
    right = geometry.vertical.second_flanks
    center = geometry.vertical.centers
    n = 2 * geometry.d
    return (
        (
            (left[0], center[1]),
            (right[n - 1], center[n - 1]),
            *((center[i], left[i]) if i % 2 else (left[i], center[i]) for i in range(2, n - 1)),
        ),
        (
            (center[1], left[1]),
            (center[n - 1], right[n - 2]),
            *((center[i], right[i - 1]) for i in range(2, n - 1)),
        ),
        ((center[1], right[0]), (center[n - 1], left[n - 1])),
        (
            (right[1], center[1]),
            *((left[i - 1], center[i]) for i in range(2, n)),
        ),
    )


def _vertical_to_horizontal_prep_map(geometry: _Geometry) -> dict[int, int]:
    """Fixed-coordinate screen-r270 map used by the exact V/H dual."""
    vertical = geometry.vertical
    horizontal = geometry.horizontal
    n = 2 * geometry.d
    return {
        **{vertical.first_flanks[i]: horizontal.second_flanks[i] for i in range(n)},
        **{vertical.second_flanks[i]: horizontal.first_flanks[i] for i in range(n)},
        **{vertical.centers[i]: horizontal.centers[i] for i in range(n + 1)},
    }


def _horizontal_prep_layers(geometry: _Geometry) -> tuple[tuple[Pair, ...], ...]:
    """Exact screen-r270/global-H dual of the selected V instrument."""
    image = _vertical_to_horizontal_prep_map(geometry)
    return tuple(
        tuple((image[target], image[control]) for control, target in layer)
        for layer in _vertical_prep_layers(geometry)
    )


def _vertical_prep_seeds(geometry: _Geometry) -> tuple[int, ...]:
    """Exact physical |+> carriers of the selected V fusion tree."""
    left = geometry.vertical.first_flanks
    right = geometry.vertical.second_flanks
    center = geometry.vertical.centers
    n = 2 * geometry.d
    return tuple(
        sorted(
            (
                *left[0:n:2],
                right[n - 1],
                *(center[i] for i in range(3, n - 1, 2)),
            )
        )
    )


def _horizontal_prep_seeds(geometry: _Geometry) -> tuple[int, ...]:
    """Physical-H targets complementary to the exact dual V carriers."""
    horizontal = geometry.horizontal
    image = _vertical_to_horizontal_prep_map(geometry)
    virtual_z_carriers = {image[qubit] for qubit in _vertical_prep_seeds(geometry)}
    return tuple(sorted(set(horizontal.resource) - virtual_z_carriers))


def _patch_layers(
    geometry: _Geometry,
    context: Literal["ZZ_PREP", "XX_PREP", "V", "H"],
) -> tuple[tuple[Pair, ...], ...]:
    """Four collision-free patch layers from the immutable axis policies."""
    if geometry.d % 2 == 0 and context in ("XX_PREP", "H"):
        return _dual_layers(
            geometry, _patch_layers(geometry, "ZZ_PREP" if context == "XX_PREP" else "V")
        )
    disabled = (
        frozenset()
        if context in ("ZZ_PREP", "XX_PREP")
        else geometry.vertical_disabled
        if context == "V"
        else geometry.horizontal_disabled
    )
    separation = 2 * geometry.d + 2
    layers: list[list[Pair]] = [[], [], [], []]
    orders = MINUS_PATCH_ORDERS if context == "H" else PLUS_PATCH_ORDERS
    patch_orders = (
        EVEN_V_PATCH_ORDERS
        if geometry.d % 2 == 0
        else tuple(orders[option] for option in PATCH_POLICY[context])
    )
    for patch_id, (x_order, z_order) in zip(PATCH_IDS, patch_orders):
        grid_x, grid_y = PATCH_OFFSETS[patch_id]
        x_offset = grid_x * separation
        y_offset = grid_y * separation
        patch = geometry.patches[patch_id]

        def add_ancilla(ancilla: Coordinate, *, is_x: bool) -> None:
            ancilla_qubit = geometry.q(ancilla)
            if ancilla_qubit in disabled:
                return
            i = (ancilla[0] - x_offset) // 2
            j = (ancilla[1] - y_offset) // 2
            order = x_order if is_x else z_order
            for layer, (dx, dy) in enumerate(order):
                data_i = i if dx == 1 else i - 1
                data_j = j if dy == 1 else j - 1
                if not (0 <= data_i < geometry.d and 0 <= data_j < geometry.d):
                    continue
                data = geometry.q(
                    (
                        2 * data_i + 1 + x_offset,
                        2 * data_j + 1 + y_offset,
                    )
                )
                pair = (ancilla_qubit, data) if is_x else (data, ancilla_qubit)
                layers[layer].append(pair)

        for ancilla in patch.x_ancillas:
            add_ancilla(ancilla, is_x=True)
        for ancilla in patch.z_ancillas:
            add_ancilla(ancilla, is_x=False)
    return tuple(tuple(layer) for layer in layers)


def _vertical_path_layers(geometry: _Geometry) -> tuple[tuple[Pair, ...], ...]:
    """Qualified V PATH colored directly around the axis-symmetric policy."""
    left = geometry.vertical.first_flanks
    right = geometry.vertical.second_flanks
    center = geometry.vertical.centers
    left_wing = geometry.vertical.first_wing
    right_wing = geometry.vertical.second_wing
    d = geometry.d
    n = 2 * d
    if d == 2:
        # Same-control contacts exchange equal-slope ticks at this endpoint.
        return (
            (
                (left[2], center[3]),
                (left[3], center[4]),
                (right[1], right_wing),
                (right[0], center[0]),
                (left[0], center[1]),
                (left[1], center[2]),
            ),
            (
                (left[0], center[0]),
                (left[2], center[2]),
                (right[3], center[4]),
                (right[2], right_wing),
                (left[1], left_wing),
            ),
            ((left[3], center[3]), (left[1], center[1])),
            ((left[2], left_wing),),
        )
    lower_start = 0 if d % 2 == 0 else 1
    return (
        (
            *((left[k], center[k + 1]) for k in range(d, n)),
            (right[d - 1], right_wing),
            (right[0], center[0]),
            *((left[k], center[k + 1]) for k in range(lower_start, d)),
            *((left[k], center[k + 1]) for k in range(0, lower_start)),
        ),
        (
            *((left[k], center[k]) for k in range(0, d)),
            *((left[k], center[k]) for k in range(d, n) if k % 2 == 0),
            (right[n - 1], center[n]),
            (right[d], right_wing),
        ),
        (
            *((left[k], center[k]) for k in range(d, n) if k % 2 == 1),
            (left[d - 1], left_wing),
        ),
        ((left[d], left_wing),),
    )


def _horizontal_path_layers(geometry: _Geometry) -> tuple[tuple[Pair, ...], ...]:
    """Qualified physical-X H PATH around the axis-symmetric policy."""
    if geometry.d % 2 == 0:
        return _dual_layers(geometry, _vertical_path_layers(geometry))
    top = geometry.horizontal.first_flanks
    bottom = geometry.horizontal.second_flanks
    center = geometry.horizontal.centers
    top_wing = geometry.horizontal.first_wing
    bottom_wing = geometry.horizontal.second_wing
    d = geometry.d
    n = 2 * d
    return (
        (
            *((center[k], top[k]) for k in range(0, n - 1)),
            (bottom_wing, bottom[d]),
            (center[n], bottom[n - 1]),
            (center[n - 1], top[n - 1]),
        ),
        (
            (center[n], top[n - 1]),
            *((center[k + 1], top[k]) for k in range(d, n - 1)),
            (center[0], bottom[0]),
            (bottom_wing, bottom[d - 1]),
            *((center[k + 1], top[k]) for k in range(d) if k % 2 == 1),
        ),
        (
            (top_wing, top[d]),
            *((center[k + 1], top[k]) for k in range(d) if k % 2 == 0),
        ),
        ((top_wing, top[d - 1]),),
    )


def _vertical_coupling(geometry: _Geometry) -> tuple[Pair, ...]:
    """Flank-control coupling to the two patch columns beside V."""
    d = geometry.d
    separation = 2 * d + 2
    pairs: list[Pair] = []
    for j, flank in enumerate(geometry.vertical.first_flanks):
        y = 2 * j + 1 if j < d else separation + 2 * (j - d) + 1
        pairs.append((flank, geometry.q((2 * d - 1, y))))
    for j, flank in enumerate(geometry.vertical.second_flanks):
        y = 2 * j + 1 if j < d else separation + 2 * (j - d) + 1
        pairs.append((flank, geometry.q((2 * d + 3, y))))
    return tuple(pairs)


def _horizontal_coupling(geometry: _Geometry) -> tuple[Pair, ...]:
    """Data-control coupling from the two patch rows beside H."""
    d = geometry.d
    separation = 2 * d + 2
    pairs: list[Pair] = []
    for j, flank in enumerate(geometry.horizontal.first_flanks):
        x = 2 * j + 1 if j < d else separation + 2 * (j - d) + 1
        pairs.append((geometry.q((x, 2 * d - 1)), flank))
    for j, flank in enumerate(geometry.horizontal.second_flanks):
        x = 2 * j + 1 if j < d else separation + 2 * (j - d) + 1
        pairs.append((geometry.q((x, 2 * d + 3)), flank))
    return tuple(pairs)


def _vertical_coupling_layers(
    geometry: _Geometry,
) -> tuple[tuple[Pair, ...], ...]:
    """Complete V coupling matching embedded directly in the pure +--+ ticks."""
    pairs = _vertical_coupling(geometry)
    if geometry.d % 2 == 0:
        # A cat-to-data CX must follow every noncommuting Z-check contact.
        # Select the last conflict-free existing layer; never add depth or
        # allow an early contact requiring a hidden patch-frame correction.
        patch = _patch_layers(geometry, "V")
        path = _vertical_path_layers(geometry)
        layers: list[list[Pair]] = [[], [], [], []]
        for pair in pairs:
            _flank, data = pair
            preceding = [
                k
                for k, layer in enumerate(patch)
                for c, t in layer
                if data == c and geometry.ancilla_basis[t] == "Z"
            ]
            legal = [
                k
                for k in range(4)
                if all(k > p for p in preceding)
                and not any(q in pair for edge in (*patch[k], *path[k]) for q in edge)
            ]
            if not legal:
                raise AssertionError(f"no legal four-layer terminal coupling for {pair}")
            layers[max(legal)].append(pair)
        return tuple(tuple(layer) for layer in layers)
    d = geometry.d
    n = 2 * d
    first, second = pairs[:n], pairs[n:]
    return (
        (),
        (
            *(first[j] for j in range(d, n) if j % 2 == 1),
            *(second[j] for j in range(0, d) if j % 2 == 0),
        ),
        (
            *(first[j] for j in range(d, n) if j % 2 == 0),
            *(second[j] for j in range(0, d) if j % 2 == 1),
        ),
        (
            *first[:d],
            *second[d:],
        ),
    )


def _horizontal_coupling_layers(
    geometry: _Geometry,
) -> tuple[tuple[Pair, ...], ...]:
    """Complete H coupling matching embedded directly in the pure -++- ticks."""
    if geometry.d % 2 == 0:
        return _dual_layers(geometry, _vertical_coupling_layers(geometry))
    pairs = _horizontal_coupling(geometry)
    d = geometry.d
    n = 2 * d
    first, second = pairs[:n], pairs[n:]
    return (
        (),
        (
            *(first[j] for j in range(0, d) if j % 2 == 0),
            *(second[j] for j in range(d, n) if j % 2 == 1),
        ),
        (
            *(first[j] for j in range(0, d) if j % 2 == 1),
            *(second[j] for j in range(d, n) if j % 2 == 0),
        ),
        (*first[d:], *second[:d]),
    )


def _validate_prep_frame(
    geometry: _Geometry,
    direction: Literal["V", "H"],
    table: dict[int, tuple[int, ...]],
) -> None:
    """Prove the fixed frame and unique null row against the live PREP tree.

    A common residual carrier on every flank is the cat's harmless global
    frame gauge; only equality of the corrected flank dependencies matters.
    """
    if direction == "V":
        corridor = geometry.vertical
        layers = _vertical_prep_layers(geometry)
        physical_h_seeds = set(_vertical_prep_seeds(geometry))
        x_cat = False
    else:
        corridor = geometry.horizontal
        layers = _horizontal_prep_layers(geometry)
        physical_h_seeds = set(_horizontal_prep_seeds(geometry))
        x_cat = True

    centers = tuple(corridor.centers[1:-1])
    active = tuple(sorted((*corridor.flanks, *centers)))
    if x_cat:
        seeds = set(active) - physical_h_seeds
        classical_layers = tuple(
            tuple((target, control) for control, target in layer) for layer in layers
        )
    else:
        seeds = physical_h_seeds
        classical_layers = layers

    seed_index = {qubit: index for index, qubit in enumerate(sorted(seeds))}
    dependency = {qubit: (1 << seed_index[qubit]) if qubit in seed_index else 0 for qubit in active}
    for layer in classical_layers:
        before = dict(dependency)
        for control, target in layer:
            if control not in dependency or target not in dependency:
                raise AssertionError("PREP tree touches a non-instrument qubit")
            dependency[target] = before[target] ^ before[control]

    basis: dict[int, int] = {}
    for center in centers:
        value = dependency[center]
        while value:
            pivot = value.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = value
                break
            value ^= basis[pivot]
    if len(basis) != len(centers) - 1:
        raise AssertionError("PREP tree does not have exactly one center null row")
    if dependency[centers[0]] ^ dependency[centers[1]]:
        raise AssertionError("PREP center null row is not the declared inner pair")
    if set(table) != set(centers):
        raise AssertionError("PREP frame table has stale center keys")

    corrected_values: set[int] = set()
    for flank in corridor.flanks:
        corrected = dependency[flank]
        for center, targets in table.items():
            if flank in targets:
                corrected ^= dependency[center]
        corrected_values.add(corrected)
    if len(corrected_values) != 1:
        raise AssertionError("PREP frame table failed its live-tree certificate")


def _vertical_x_frame_table(geometry: _Geometry) -> dict[int, tuple[int, ...]]:
    """Exact direct X-frame dependence for the centered V preparation."""
    left = geometry.vertical.first_flanks
    right = geometry.vertical.second_flanks
    center = geometry.vertical.centers
    n = 2 * geometry.d
    result = {
        center[1]: tuple(sorted((*left[2:n], *right[1:n]))),
        center[2]: (),
    }
    for index in range(3, n):
        result[center[index]] = tuple(sorted((*left[index:n], *right[index - 1 : n])))
    return result


def _horizontal_z_frame_table(geometry: _Geometry) -> dict[int, tuple[int, ...]]:
    """Exact screen-r270/global-H dual frame of the V preparation."""
    top = geometry.horizontal.first_flanks
    bottom = geometry.horizontal.second_flanks
    center = geometry.horizontal.centers
    n = 2 * geometry.d
    result = {
        center[1]: tuple(sorted((*bottom[2:n], *top[1:n]))),
        center[2]: (),
    }
    for index in range(3, n):
        result[center[index]] = tuple(sorted((*bottom[index:n], *top[index - 1 : n])))
    return result


# =============================================================================
# Named detector bookkeeping
# =============================================================================


@dataclass(frozen=True)
class _MeasurementRecord:
    name: str
    absolute_index: int
    qubit: int


class _RecordBook:
    """Assign stable semantic names at the instant measurements are emitted."""

    def __init__(self) -> None:
        self._records: dict[str, _MeasurementRecord] = {}
        self._last_by_qubit: dict[int, _MeasurementRecord] = {}
        self.measurement_count = 0

    def add(self, name: str, qubit: int) -> _MeasurementRecord:
        if name in self._records:
            raise ValueError(f"duplicate measurement role {name!r}")
        record = _MeasurementRecord(name, self.measurement_count, qubit)
        self._records[name] = record
        self._last_by_qubit[qubit] = record
        self.measurement_count += 1
        return record

    def discard(self, qubit: int) -> None:
        """Advance past an intentionally ignored physical measurement.

        A refresh measurement exists in Stim's global measurement stream, so
        it must advance every later ``rec[]`` offset.  It is deliberately not
        assigned a semantic name and does not replace ``last(qubit)``: frames,
        detectors, and observables must continue to refer to the qubit's last
        experiment measurement, never to this discarded refresh outcome.
        """
        if not isinstance(qubit, int):
            raise TypeError("discarded measurement target must be a qubit index")
        self.measurement_count += 1

    def record(self, name: str) -> _MeasurementRecord:
        return self._records[name]

    def last(self, qubit: int) -> _MeasurementRecord:
        return self._last_by_qubit[qubit]

    def rec_target(self, record: _MeasurementRecord) -> str:
        return f"rec[-{self.measurement_count - record.absolute_index}]"


# Detector coordinates use a documented fourth coordinate instead of magic
# integers.  Their final numeric values remain part of the DEM contract.
TAG_LOCAL = 0
TAG_VERTICAL_OUTER_PARITY = 1
TAG_HORIZONTAL_OUTER_PARITY = 2
TAG_GLOBAL_CLOSURE_PARITY = 3
TAG_CORRIDOR_PREP_RELATION = 4


DetectorCoordinate = tuple[int, ...]


def _detector_basis_block(
    *,
    d: int,
    rv: int,
    start: int,
    direction: Literal["V", "H"],
    first: bool,
    final: bool,
    memory: Memory,
) -> dict[DetectorCoordinate, frozenset[DetectorCoordinate]]:
    """Return the bounded detector-basis rows owned by one corridor block.

    Each row names atomic syndrome parities available at one chronological
    declaration.  The construction depends only on the local V/H block role,
    never on ``ro`` or a global list of rounds.  Only the final named-record
    parity is emitted; there is no circuit rewrite or post-processing pass.
    """
    if direction == "H" and d % 2 == 0:
        # The balanced H instrument is the exact spatial/Pauli dual of V,
        # including detector parents, outer parity and preparation basis.
        rows = _detector_basis_block(
            d=d,
            rv=rv,
            start=start,
            direction="V",
            first=first,
            final=final,
            memory="XX" if memory == "ZZ" else "ZZ",
        )

        def dual(coordinate: DetectorCoordinate) -> DetectorCoordinate:
            x, y, time, *tags = coordinate
            return (y, 4 * d + 2 - x, time) + tuple(
                3 - tag if tag in (TAG_VERTICAL_OUTER_PARITY, TAG_HORIZONTAL_OUTER_PARITY) else tag
                for tag in tags
            )

        return {dual(target): frozenset(map(dual, support)) for target, support in rows.items()}
    center = 2 * d + 1
    horizontal = direction == "H"
    readout_corridor: Literal["V", "H"] = "V" if _is_x_memory(memory) else "H"
    same_corridor_tail = final and direction == readout_corridor
    rows: dict[DetectorCoordinate, frozenset[DetectorCoordinate]] = {}

    def local(
        dx: int,
        dy: int,
        time: int,
        tag: int | None = None,
    ) -> DetectorCoordinate:
        if horizontal:
            dx, dy = -dy, dx
        coordinate: DetectorCoordinate = (center + dx, center + dy, time)
        if tag is not None:
            coordinate += (tag,)
        return coordinate

    def child(dx: int, dy: int, round_index: int) -> DetectorCoordinate:
        return local(dx, dy, start + round_index)

    def set_row(
        target: DetectorCoordinate,
        support: Iterable[DetectorCoordinate],
    ) -> None:
        if target in rows:
            raise AssertionError(f"duplicate detector-basis row {target}")
        value = frozenset(support)
        if not value:
            raise AssertionError("detector-basis row cannot be empty")
        rows[target] = value

    # Close the ordinary verifier chains into the terminal patch syndromes.
    parent_time = start + rv + (1 if final else 0)
    ordinary_offsets: list[int] = []
    for k in range(1, d):
        y = -2 * k
        x = -3 if k % 2 else -1
        ordinary_offsets.append(y)
        target = local(x, y - 1, parent_time)
        support = {target, child(0, y, 0)}
        if rv > 1:
            support.add(child(0, y, rv - 1))
        if not same_corridor_tail:
            set_row(target, support)
    for k in range(1, d - 1):
        y = 2 * k
        x = -1 if k % 2 else -3
        ordinary_offsets.append(y)
        target = local(x, y + 1, parent_time)
        support = {target, child(0, y, 0)}
        if rv > 1:
            support.add(child(0, y, rv - 1))
        if not same_corridor_tail:
            set_row(target, support)
    y = 2 * d - 2
    k = d - 1
    x = -1 if k % 2 else -3
    target = local(x, y + 1, parent_time)
    support = {target, child(0, y, 0)}
    if rv > 1:
        support.add(child(0, y, rv - 1))
    if not same_corridor_tail:
        set_row(target, support)

    top = -2 * d
    for round_index in range(2, rv):
        for offset in (top, *ordinary_offsets):
            set_row(
                child(0, offset, round_index),
                {
                    child(0, offset, round_index - 1),
                    child(0, offset, round_index),
                },
            )

    # Resolve the three exceptional verifier chains at the stitch center.
    def middle(k):
        return child(0, 0, k)

    def left(k):
        return child(-2, 0, k)

    def right(k):
        return child(2, 0, k)

    set_row(left(0), {left(0), middle(0)})
    if rv >= 2:
        set_row(middle(1), {middle(1), middle(0), left(0)})
        set_row(left(1), {left(1), middle(1), middle(0), left(0)})
    for round_index in range(2, rv):
        set_row(
            middle(round_index),
            {middle(round_index), middle(0), left(0), left(round_index - 1)},
        )
        set_row(
            left(round_index),
            {left(round_index), middle(round_index), middle(0), left(0)},
        )
        set_row(
            right(round_index),
            {right(round_index - 1), right(round_index)},
        )

    # Resolve the far corridor boundary, including the rv=2 endpoint case.
    def adjacent(k):
        return child(0, 2 * d - 2, k)

    def boundary(k):
        return child(0, 2 * d, k)

    set_row(boundary(0), {boundary(0)})
    if rv >= 2:
        if rv == 2:
            set_row(adjacent(1), {adjacent(1), boundary(0)})
            set_row(boundary(1), {boundary(1), boundary(0)})
        else:
            set_row(adjacent(1), {adjacent(1)})
            set_row(boundary(1), {boundary(1)})
            for round_index in range(2, rv):
                set_row(
                    adjacent(round_index),
                    {adjacent(round_index - 1), adjacent(round_index)},
                )
                set_row(
                    boundary(round_index),
                    {boundary(round_index - 1), boundary(round_index)},
                )

    # Close the module's deterministic outer parity at its physical endpoint.
    # The next opposite corridor can be its first destructive measurement.
    # In that case no outer detector exists to own a basis row.
    if first and not final and direction == readout_corridor:
        return rows
    if final:
        outer_time, outer_tag = start + rv + 1, TAG_GLOBAL_CLOSURE_PARITY
    elif horizontal:
        outer_time, outer_tag = start + 2 * rv, TAG_VERTICAL_OUTER_PARITY
    else:
        outer_time, outer_tag = start + 2 * rv, TAG_HORIZONTAL_OUTER_PARITY
    outer = local(0, 0, outer_time, outer_tag)
    if same_corridor_tail:
        # If readout closes the same corridor as the final block, the raw
        # flank XOR data row is already the complete endpoint relation.
        # Cross-corridor closure alone needs the PATH telescope below.
        set_row(outer, {outer})
    else:
        outer_support = {outer, left(0), right(0)}
        if rv > 1:
            outer_support.update((left(rv - 1), right(rv - 1)))
        set_row(outer, outer_support)

    return rows


# =============================================================================
# Deterministic instruction and measurement primitives
# =============================================================================


class _StimSource:
    """Append deterministic Stim text; parse only once at finalization."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def append(
        self,
        name: str,
        targets: Sequence[int | str] = (),
        argument: float | Sequence[float] | None = None,
    ) -> None:
        suffix = ""
        if argument is not None:
            values = (argument,) if isinstance(argument, (int, float)) else argument
            suffix = "(" + ", ".join(format(float(v), ".17g") for v in values) + ")"
        target_text = " ".join(map(str, targets))
        self.lines.append(f"{name}{suffix}" + (f" {target_text}" if target_text else ""))

    def parse(self) -> stim.Circuit:
        return stim.Circuit("\n".join(self.lines))


@dataclass(frozen=True)
class _MeasurementGroup:
    qubits: tuple[int, ...]
    basis: Basis
    role_prefix: str


def _flat_pairs(pairs: Sequence[Pair]) -> tuple[int, ...]:
    return tuple(qubit for pair in pairs for qubit in pair)


# =============================================================================
# Chronological emission: PREP, uniform V/H modules, and DATA
# =============================================================================


class _ChronologicalWriter:
    """Own all physical and detector state while emitting strictly forward."""

    def __init__(
        self,
        *,
        d: int,
        rv: int,
        memory: Memory,
    ):
        self.geometry = _build_geometry(d)
        self.rv = rv
        self.memory = memory
        self.source = _StimSource()
        self.records = _RecordBook()
        self._patch_layers_by_context = {
            context: _patch_layers(self.geometry, context)
            for context in ("ZZ_PREP", "XX_PREP", "V", "H")
        }
        self._vertical_prep = _vertical_prep_layers(self.geometry)
        self._horizontal_prep = _horizontal_prep_layers(self.geometry)
        self._vertical_path = _vertical_path_layers(self.geometry)
        self._horizontal_path = _horizontal_path_layers(self.geometry)
        self._vertical_coupling = _vertical_coupling(self.geometry)
        self._horizontal_coupling = _horizontal_coupling(self.geometry)
        self._vertical_coupling_schedule = _vertical_coupling_layers(self.geometry)
        self._horizontal_coupling_schedule = _horizontal_coupling_layers(self.geometry)
        self._vertical_frame_table = _vertical_x_frame_table(self.geometry)
        self._horizontal_frame_table = _horizontal_z_frame_table(self.geometry)
        _validate_prep_frame(self.geometry, "V", self._vertical_frame_table)
        _validate_prep_frame(self.geometry, "H", self._horizontal_frame_table)
        self.aliases: dict[str, frozenset[str]] = {}
        # Every ordinary detector is a deterministic GF(2) relation.  Orient
        # that relation from its oldest record toward newer boundary records
        # as soon as it is declared.  The final logical representatives are
        # reduced through these causal rules, so one bounded frame reaches the
        # endpoint instead of accumulated preparation records or feedback.
        self._forward_frame_rules: dict[str, frozenset[str]] = {}
        self.data_x: dict[int, frozenset[str]] = {q: frozenset() for q in self.geometry.data}
        self.data_z: dict[int, frozenset[str]] = {q: frozenset() for q in self.geometry.data}
        self.time = 0
        self.detector_count = 0
        self._atomic_detectors: dict[DetectorCoordinate, frozenset[str]] = {}
        self._detector_basis_rows: dict[
            DetectorCoordinate,
            frozenset[DetectorCoordinate],
        ] = {}
        self._basis_coordinates: set[DetectorCoordinate] = set()
        self._patch_reference: dict[int, frozenset[str]] = {}
        self._verify_anchor: dict[str, dict[int, frozenset[str]]] = {}
        self._previous_coupling: dict[str, dict[int, frozenset[str]]] = {
            "V": {},
            "H": {},
        }
        self._last_module: Literal["V", "H"] | None = None
        self._data_physically_measured = False

    def _x_ancillas(self, disabled: frozenset[int] = frozenset()) -> tuple[int, ...]:
        return tuple(
            sorted(
                self.geometry.q(coordinate)
                for patch in self.geometry.patches.values()
                for coordinate in patch.x_ancillas
                if self.geometry.q(coordinate) not in disabled
            )
        )

    def _active_patch_ancillas(self, disabled: frozenset[int]) -> tuple[int, ...]:
        return tuple(q for q in self.geometry.ancillas if q not in disabled)

    def _measurement_groups(
        self,
        groups: Sequence[_MeasurementGroup],
    ) -> list[tuple[tuple[int, ...], Basis, dict[int, str]]]:
        """Validate disjoint targets and merge one layer by measurement basis."""
        seen: set[int] = set()
        merged: dict[Basis, dict[int, str]] = {}
        for group in groups:
            targets = set(group.qubits)
            if len(targets) != len(group.qubits) or seen & targets:
                raise AssertionError("duplicate measurement targets")
            seen.update(targets)
            roles = merged.setdefault(group.basis, {})
            roles.update({q: f"{group.role_prefix}.q{q}" for q in targets})
        return [(tuple(sorted(roles)), basis, roles) for basis, roles in merged.items()]

    def _emit_sweep_prep(self, *, data_basis: Basis) -> None:
        """Prepare data from independent sweep bits and implicit |0> starts.

        Stim qubits begin in |0>.  A classically controlled X therefore loads
        one configurable computational-basis bit into each data qubit.  An H
        layer converts those configured Z eigenstates into X eigenstates when
        requested.  Ancillas intentionally receive no initial reset in this
        mode; their implicit |0> initialization is the preparation.
        """
        sweep_targets: list[int | str] = []
        for sweep_index, qubit in enumerate(self.geometry.data):
            sweep_targets.extend((f"sweep[{sweep_index}]", qubit))
        self.source.append("CX", sweep_targets)
        if data_basis == "X":
            self.source.append("TICK")
            self.source.append("H", self.geometry.data)

    def _emit_cycle(
        self,
        *,
        patch: tuple[tuple[Pair, ...], ...],
        x_ancillas: tuple[int, ...],
        corridor: tuple[tuple[Pair, ...], ...] | None = None,
        corridor_seed_h: tuple[int, ...] = (),
        merge_leading_h_with: tuple[int, ...] = (),
        merge_trailing_h_with: tuple[int, ...] = (),
    ) -> None:
        """Emit one four-CX-layer stabilizer/corridor cycle."""
        opening_h = tuple((*x_ancillas, *corridor_seed_h))
        if merge_leading_h_with:
            overlap = set(opening_h) & set(merge_leading_h_with)
            if overlap:
                raise AssertionError(
                    f"overlapping leading H layers cannot be merged: {sorted(overlap)}"
                )
        else:
            self.source.append("TICK")
        self.source.append("H", opening_h)
        self.source.append("TICK")
        for layer in range(4):
            patch_targets = _flat_pairs(patch[layer])
            corridor_targets = _flat_pairs(corridor[layer]) if corridor else ()
            if patch_targets or corridor_targets:
                self.source.append("CX", (*patch_targets, *corridor_targets))
            self.source.append("TICK")
        self.source.append("H", x_ancillas)
        if merge_trailing_h_with:
            overlap = set(x_ancillas) & set(merge_trailing_h_with)
            if overlap:
                raise AssertionError(
                    f"overlapping trailing H layers cannot be merged: {sorted(overlap)}"
                )
        else:
            self.source.append("TICK")

    def _emit_terminal_cx_layers(
        self,
        *,
        patch: tuple[tuple[Pair, ...], ...],
        path: tuple[tuple[Pair, ...], ...],
        coupling: tuple[tuple[Pair, ...], ...],
    ) -> None:
        """Emit the four collision-free terminal interaction layers."""
        for tick in range(4):
            patch_targets = _flat_pairs(patch[tick])
            corridor_targets = _flat_pairs((*path[tick], *coupling[tick]))
            all_targets = (*patch_targets, *corridor_targets)
            if all_targets:
                self.source.append("CX", all_targets)
            self.source.append("TICK")

    def _emit_measure_reset_layers(
        self,
        groups: Sequence[_MeasurementGroup],
        *,
        ghz_frame_qubits: Sequence[int] = (),
        post_reset_h_qubits: Sequence[int] = (),
        restore_x: bool = True,
        close_data: bool = False,
        close_data_h_prepared: bool = False,
    ) -> None:
        """Measure semantic records, discard idle refreshes, then reset.

        At a nonfinal cycle boundary, refresh every non-data qubit except a
        live GHZ carrier. A dummy outcome advances the physical record count
        without replacing the qubit's last semantic record. Final data and
        terminal ancillas share one M moment and have no following reset.
        """
        normalized = self._measurement_groups(groups)
        measured = {q for targets, _basis, _roles in normalized for q in targets}
        non_data = set(self.geometry.qubit_to_coordinate) - set(self.geometry.data)
        if measured - non_data:
            raise AssertionError("semantic ancilla layer contains data")
        live_frame = set(ghz_frame_qubits)
        if live_frame - non_data or live_frame & measured:
            raise AssertionError("invalid or measured live GHZ carrier")
        if close_data and live_frame:
            raise AssertionError("final readout cannot preserve a future GHZ frame")
        refresh = () if close_data else tuple(sorted(non_data - measured - live_frame))

        x_targets = tuple(
            sorted(q for targets, basis, _roles in normalized if basis == "X" for q in targets)
        )
        if x_targets:
            self.source.append("H", x_targets)
            if close_data and _is_x_memory(self.memory) and not close_data_h_prepared:
                self.source.append("H", self.geometry.data)
                close_data_h_prepared = True
            self.source.append("TICK")
        elif close_data and _is_x_memory(self.memory) and not close_data_h_prepared:
            self.source.append("H", self.geometry.data)
            self.source.append("TICK")

        # Semantic targets first, anonymous refreshes second. Preserve the
        # literal p=0 baseline ordering, not merely the target sets.
        measurement_targets = tuple(sorted(measured))
        roles = {q: role for _targets, _basis, mapping in normalized for q, role in mapping.items()}
        all_targets = (*measurement_targets, *refresh)
        if all_targets:
            self.source.append("M", all_targets)
            for q in measurement_targets:
                self.records.add(roles[q], q)
            for q in refresh:
                self.records.discard(q)
        if close_data:
            if self._data_physically_measured:
                raise AssertionError("data readout was emitted twice")
            self.source.append("M", self.geometry.data)
            for q in self.geometry.data:
                self.records.add(f"data.q{q}", q)
            self._data_physically_measured = True
        elif all_targets:
            self.source.append("TICK")
            self.source.append("R", all_targets)

        post_reset_h = set(post_reset_h_qubits)
        if post_reset_h - set(all_targets):
            raise AssertionError("post-reset H target was not reset")
        if close_data and post_reset_h:
            raise AssertionError("post-reset H requested after final measurement")
        restore_targets = (set(x_targets) if restore_x else set()) | post_reset_h
        if restore_targets:
            self.source.append("TICK")
            self.source.append("H", tuple(sorted(restore_targets)))

    @staticmethod
    def _xor(parities: Iterable[frozenset[str]]) -> frozenset[str]:
        result: frozenset[str] = frozenset()
        for parity in parities:
            result ^= parity
        return result

    def _record_parity(self, record: _MeasurementRecord) -> frozenset[str]:
        # Detector construction uses the qualified raw causal ledger.  The
        # separate forward quotient is applied only to logical representatives
        # so detector locality and strict graphlike decomposition are retained.
        return self.aliases.get(record.name, frozenset((record.name,)))

    def _canonicalize_frame(self, parity: Iterable[str]) -> frozenset[str]:
        """Push a symbolic parity through already-declared causal relations."""
        pending = set(parity)
        result: frozenset[str] = frozenset()
        while pending:
            name = pending.pop()
            replacement = self._forward_frame_rules.get(name)
            if replacement is None:
                result ^= frozenset((name,))
            else:
                # Symmetric difference cancels duplicate paths immediately;
                # a list expansion can grow exponentially at large RO.
                pending.symmetric_difference_update(replacement)
        return result

    def _register_forward_relation(self, parity: Iterable[str]) -> None:
        """Use one detector to replace its oldest record by newer records."""
        canonical = self._canonicalize_frame(parity)
        if not canonical:
            raise AssertionError("dependent forward frame relation")
        records = sorted(
            (self.records.record(name) for name in canonical),
            key=lambda record: record.absolute_index,
        )
        pivot = records[0]
        replacement = canonical ^ frozenset((pivot.name,))
        if any(
            self.records.record(name).absolute_index <= pivot.absolute_index for name in replacement
        ):
            raise AssertionError("frame relation did not point forward")
        self._forward_frame_rules[pivot.name] = replacement

    def _set_last_alias(
        self,
        qubit: int,
        extra_terms: Iterable[frozenset[str]],
    ) -> None:
        record = self.records.last(qubit)
        self.aliases[record.name] = frozenset((record.name,)) ^ self._xor(extra_terms)

    def _corridor_frame_from_centers(
        self,
        center_to_flanks: dict[int, tuple[int, ...]],
    ) -> dict[int, frozenset[str]]:
        """Invert a fixed center-to-flank table into explicit frame parities."""
        result: dict[int, frozenset[str]] = {}
        for center, flanks in center_to_flanks.items():
            center_parity = self._record_parity(self.records.last(center))
            for flank in flanks:
                result[flank] = result.get(flank, frozenset()) ^ center_parity
        return result

    def _alias_patch_layer(self, measured_ancillas: Sequence[int]) -> None:
        """Attach only the appropriate data frame to each patch-check result."""
        for ancilla in measured_ancillas:
            data_frame = self.data_z if self.geometry.ancilla_basis[ancilla] == "X" else self.data_x
            self._set_last_alias(
                ancilla,
                (data_frame[data] for data in self.geometry.ancilla_data_neighbors[ancilla]),
            )

    def _alias_path_layer(
        self,
        direction: Literal["V", "H"],
        path: tuple[tuple[Pair, ...], ...],
        flank_frame: dict[int, frozenset[str]],
    ) -> None:
        incident: dict[int, list[int]] = {}
        for control, target in (pair for layer in path for pair in layer):
            check, flank = (target, control) if direction == "V" else (control, target)
            incident.setdefault(check, []).append(flank)
        for check, flanks in incident.items():
            self._set_last_alias(
                check,
                (flank_frame.get(flank, frozenset()) for flank in flanks),
            )

    def _alias_vertical_terminal_flanks(self) -> None:
        for flank, data in self._vertical_coupling:
            self._set_last_alias(flank, (self.data_z[data],))

    def _alias_horizontal_terminal_flanks(self) -> None:
        for data, flank in self._horizontal_coupling:
            self._set_last_alias(flank, (self.data_x[data],))

    def _register_detector_basis_block(
        self,
        direction: Literal["V", "H"],
        *,
        final: bool,
    ) -> None:
        """Register one local block before any of its records are emitted."""
        rows = _detector_basis_block(
            d=self.geometry.d,
            rv=self.rv,
            start=self.time + 1,
            direction=direction,
            first=self._last_module is None,
            final=final,
            memory=self.memory,
        )
        overlap = set(rows) & set(self._detector_basis_rows)
        if overlap:
            raise AssertionError(f"duplicate detector-basis rows: {sorted(overlap)}")
        self._detector_basis_rows.update(rows)
        self._basis_coordinates.update(rows)
        for support in rows.values():
            self._basis_coordinates.update(support)

    def emit_prep(self) -> None:
        x_memory = _is_x_memory(self.memory)
        marker = "X_PREP" if x_memory else "Z_PREP"
        self.source.lines.append(f"# MODULE_BEGIN {marker}")
        for qubit in sorted(self.geometry.qubit_to_coordinate):
            x, y = self.geometry.qubit_to_coordinate[qubit]
            self.source.append("QUBIT_COORDS", (qubit,), (x, y))

        data_basis: Basis = "X" if x_memory else "Z"
        self._emit_sweep_prep(data_basis=data_basis)

        self._emit_cycle(
            patch=self._patch_layers_by_context[
                "ZZ_PREP" if _uses_zz_flow(self.memory) else "XX_PREP"
            ],
            x_ancillas=self._x_ancillas(),
            merge_leading_h_with=self.geometry.data if x_memory else (),
        )
        self._emit_measure_reset_layers(
            (
                _MeasurementGroup(
                    self.geometry.ancillas,
                    "Z",
                    "prep.patch",
                ),
            ),
        )
        self._alias_patch_layer(self.geometry.ancillas)
        self._emit_initial_patch_detectors()
        self.source.lines.append(f"# MODULE_END {marker}")

    def emit_vertical_module(self, module_name: str, *, final: bool = False) -> None:
        self._register_detector_basis_block("V", final=final)
        self.source.lines.append("# MODULE_BEGIN V")
        geometry = self.geometry
        corridor = geometry.vertical
        active_patch = self._active_patch_ancillas(geometry.vertical_disabled)
        x_ancillas = self._x_ancillas(geometry.vertical_disabled)
        seed_h = _vertical_prep_seeds(geometry)

        self._emit_cycle(
            patch=self._patch_layers_by_context["V"],
            x_ancillas=x_ancillas,
            corridor=self._vertical_prep,
            corridor_seed_h=seed_h,
        )
        self._emit_measure_reset_layers(
            (
                _MeasurementGroup(
                    tuple(sorted(corridor.centers[1:-1])),
                    "Z",
                    f"{module_name}.prep.center",
                ),
                _MeasurementGroup(
                    active_patch,
                    "Z",
                    f"{module_name}.prep.patch",
                ),
            ),
            ghz_frame_qubits=corridor.flanks,
        )
        self._alias_patch_layer(active_patch)
        flank_x = self._corridor_frame_from_centers(
            self._vertical_frame_table,
        )
        self._emit_module_prep_detectors("V", module_name, active_patch)

        verify_targets = tuple(sorted((*corridor.centers, *corridor.flags)))
        path = self._vertical_path
        for round_index in range(self.rv):
            round_name = f"{module_name}.verify{round_index}"
            if round_index + 1 < self.rv:
                self._emit_cycle(
                    patch=self._patch_layers_by_context["V"],
                    x_ancillas=x_ancillas,
                    corridor=path,
                )
                self._emit_measure_reset_layers(
                    (
                        _MeasurementGroup(verify_targets, "Z", f"{round_name}.path"),
                        _MeasurementGroup(active_patch, "Z", f"{round_name}.patch"),
                    ),
                    ghz_frame_qubits=corridor.flanks,
                )
            else:
                self._emit_vertical_terminal(
                    round_name=round_name,
                    active_patch=active_patch,
                    x_ancillas=x_ancillas,
                    path=path,
                    close_data=final,
                )
            self._alias_path_layer("V", path, flank_x)
            if round_index + 1 == self.rv:
                self._alias_vertical_terminal_flanks()
            self._alias_patch_layer(active_patch)
            self._emit_verify_detectors("V", module_name, round_index, verify_targets, active_patch)

        for flank, data in self._vertical_coupling:
            self.data_x[data] ^= flank_x.get(flank, frozenset())
        self.source.lines.append("# MODULE_END V")

    def emit_horizontal_module(self, module_name: str, *, final: bool = False) -> None:
        self._register_detector_basis_block("H", final=final)
        self.source.lines.append("# MODULE_BEGIN H")
        geometry = self.geometry
        corridor = geometry.horizontal
        active_patch = self._active_patch_ancillas(geometry.horizontal_disabled)
        x_ancillas = self._x_ancillas(geometry.horizontal_disabled)
        seed_h = _horizontal_prep_seeds(geometry)
        prep_measure_x = tuple(sorted(corridor.centers[1:-1]))

        self._emit_cycle(
            patch=self._patch_layers_by_context["H"],
            x_ancillas=x_ancillas,
            corridor=self._horizontal_prep,
            corridor_seed_h=seed_h,
            merge_trailing_h_with=prep_measure_x,
        )
        self._emit_measure_reset_layers(
            (
                _MeasurementGroup(
                    prep_measure_x,
                    "X",
                    f"{module_name}.prep.center",
                ),
                _MeasurementGroup(
                    active_patch,
                    "Z",
                    f"{module_name}.prep.patch",
                ),
            ),
            ghz_frame_qubits=corridor.flanks,
            post_reset_h_qubits=tuple(sorted(set(seed_h) - set(corridor.flanks))),
        )
        self._alias_patch_layer(active_patch)
        flank_z = self._corridor_frame_from_centers(
            self._horizontal_frame_table,
        )
        self._emit_module_prep_detectors("H", module_name, active_patch)

        verify_targets = tuple(sorted((*corridor.centers, *corridor.flags)))
        path = self._horizontal_path
        for round_index in range(self.rv):
            round_name = f"{module_name}.verify{round_index}"
            if round_index + 1 < self.rv:
                self._emit_cycle(
                    patch=self._patch_layers_by_context["H"],
                    x_ancillas=x_ancillas,
                    corridor=path,
                    merge_leading_h_with=verify_targets,
                    merge_trailing_h_with=verify_targets,
                )
                self._emit_measure_reset_layers(
                    (
                        _MeasurementGroup(verify_targets, "X", f"{round_name}.path"),
                        _MeasurementGroup(active_patch, "Z", f"{round_name}.patch"),
                    ),
                    ghz_frame_qubits=corridor.flanks,
                )
            else:
                self._emit_horizontal_terminal(
                    round_name=round_name,
                    active_patch=active_patch,
                    x_ancillas=x_ancillas,
                    path=path,
                    merge_leading_h_with=verify_targets,
                    merge_trailing_h_with=verify_targets,
                    close_data=final,
                )
            self._alias_path_layer("H", path, flank_z)
            if round_index + 1 == self.rv:
                self._alias_horizontal_terminal_flanks()
            self._alias_patch_layer(active_patch)
            self._emit_verify_detectors("H", module_name, round_index, verify_targets, active_patch)

        for data, flank in self._horizontal_coupling:
            self.data_z[data] ^= flank_z.get(flank, frozenset())
        self.source.lines.append("# MODULE_END H")

    def emit_data_readout(self) -> None:
        if not self._data_physically_measured:
            raise AssertionError("final module did not physically measure data")
        x_memory = _is_x_memory(self.memory)
        marker = "DATA_M_X" if x_memory else "DATA_M_Z"
        self.source.lines.append(f"# MODULE_BEGIN {marker}")
        basis: Basis = "X" if x_memory else "Z"
        frame = self.data_z if basis == "X" else self.data_x
        for qubit in self.geometry.data:
            record = self.records.last(qubit)
            self.aliases[record.name] = frozenset((record.name,)) ^ frame[qubit]
        self._emit_tail_detectors_and_observables()
        self.source.lines.append(f"# MODULE_END {marker}")

    def _emit_vertical_terminal(
        self,
        *,
        round_name: str,
        active_patch: tuple[int, ...],
        x_ancillas: tuple[int, ...],
        path: tuple[tuple[Pair, ...], ...],
        close_data: bool,
    ) -> None:
        self.source.append("TICK")
        self.source.append("H", x_ancillas)
        self.source.append("TICK")
        self._emit_terminal_cx_layers(
            patch=self._patch_layers_by_context["V"],
            path=path,
            coupling=self._vertical_coupling_schedule,
        )
        flanks = self.geometry.vertical.flanks
        self.source.append("H", (*x_ancillas, *flanks))
        close_data_h_prepared = close_data and _is_x_memory(self.memory)
        if close_data_h_prepared:
            self.source.append("H", self.geometry.data)
        self.source.append("TICK")
        self._emit_measure_reset_layers(
            (
                _MeasurementGroup(
                    tuple(sorted((*self.geometry.vertical.centers, *self.geometry.vertical.flags))),
                    "Z",
                    f"{round_name}.path",
                ),
                _MeasurementGroup(flanks, "Z", f"{round_name}.flank"),
                _MeasurementGroup(active_patch, "Z", f"{round_name}.patch"),
            ),
            close_data=close_data,
            close_data_h_prepared=close_data_h_prepared,
        )

    def _emit_horizontal_terminal(
        self,
        *,
        round_name: str,
        active_patch: tuple[int, ...],
        x_ancillas: tuple[int, ...],
        path: tuple[tuple[Pair, ...], ...],
        merge_leading_h_with: tuple[int, ...],
        merge_trailing_h_with: tuple[int, ...],
        close_data: bool,
    ) -> None:
        overlap = set(x_ancillas) & set(merge_leading_h_with)
        if overlap:
            raise AssertionError(
                f"overlapping terminal leading H layers cannot be merged: {sorted(overlap)}"
            )
        self.source.append("H", x_ancillas)
        self.source.append("TICK")
        self._emit_terminal_cx_layers(
            patch=self._patch_layers_by_context["H"],
            path=path,
            coupling=self._horizontal_coupling_schedule,
        )
        self.source.append("H", x_ancillas)
        overlap = set(x_ancillas) & set(merge_trailing_h_with)
        if overlap:
            raise AssertionError(
                f"overlapping terminal trailing H layers cannot be merged: {sorted(overlap)}"
            )
        self._emit_measure_reset_layers(
            (
                _MeasurementGroup(
                    tuple(
                        sorted((*self.geometry.horizontal.centers, *self.geometry.horizontal.flags))
                    ),
                    "X",
                    f"{round_name}.path",
                ),
                _MeasurementGroup(
                    self.geometry.horizontal.flanks,
                    "Z",
                    f"{round_name}.flank",
                ),
                _MeasurementGroup(active_patch, "Z", f"{round_name}.patch"),
            ),
            restore_x=False,
            close_data=close_data,
        )

    def _emit_named_detector(
        self,
        *,
        coordinate: DetectorCoordinate,
        coordinate_arguments: Sequence[float],
        atomic_parity: frozenset[str],
    ) -> None:
        """Emit one detector from its bounded causal basis row."""
        # Ordinary patch rows can intentionally reuse the same coordinate at
        # one timestamp (HEAD and the first module PREP).  Coordinate keys are
        # reserved only for central PATH basis rows, whose formula guarantees
        # uniqueness.  All other rows are emitted from their semantic callsite.
        if coordinate in self._basis_coordinates:
            existing = self._atomic_detectors.get(coordinate)
            if existing is not None and existing != atomic_parity:
                raise AssertionError(f"conflicting detector-basis coordinate {coordinate}")
            self._atomic_detectors[coordinate] = atomic_parity
        basis_row = self._detector_basis_rows.get(
            coordinate,
            frozenset((coordinate,)),
        )
        parity = (
            self._xor(self._atomic_detectors[source] for source in basis_row)
            if coordinate in self._detector_basis_rows
            else atomic_parity
        )
        if not parity:
            raise AssertionError(f"empty detector parity at {coordinate}")
        ordered_records = sorted(
            (self.records.record(name) for name in parity),
            key=lambda record: record.absolute_index,
        )
        self.source.append(
            "DETECTOR",
            tuple(self.records.rec_target(record) for record in ordered_records),
            coordinate_arguments,
        )
        self.detector_count += 1
        self._register_forward_relation(parity)

    def _shift_time(self) -> None:
        self.source.append("SHIFT_COORDS", (), (0, 0, 1))
        self.time += 1

    def _patch_basis(self, qubit: int) -> Basis:
        try:
            return self.geometry.ancilla_basis[qubit]
        except KeyError as error:
            raise ValueError(f"qubit {qubit} is not a patch ancilla") from error

    def _emit_initial_patch_detectors(self) -> None:
        compatible: Basis = "X" if _is_x_memory(self.memory) else "Z"
        for qubit in self.geometry.ancillas:
            parity = self._record_parity(self.records.last(qubit))
            self._patch_reference[qubit] = parity
            if self._patch_basis(qubit) != compatible:
                continue
            x, y = self.geometry.qubit_to_coordinate[qubit]
            self._emit_named_detector(
                coordinate=(x, y, self.time),
                coordinate_arguments=(x, y, 0),
                atomic_parity=parity,
            )

    def _emit_module_prep_detectors(
        self,
        direction: Literal["V", "H"],
        module_name: str,
        active_patch: tuple[int, ...],
    ) -> None:
        detector_qubits = set(active_patch)
        if direction == "V" and self._last_module == "H":
            detector_qubits |= set(self.geometry.horizontal_disabled)
        if direction == "H":
            detector_qubits |= set(self.geometry.vertical_disabled)
        for qubit in sorted(detector_qubits):
            current = self._record_parity(self.records.last(qubit))
            atomic = current ^ self._patch_reference[qubit]
            x, y = self.geometry.qubit_to_coordinate[qubit]
            self._emit_named_detector(
                coordinate=(x, y, self.time),
                coordinate_arguments=(x, y, 0),
                atomic_parity=atomic,
            )
            self._patch_reference[qubit] = current
        self._emit_corridor_prep_relation(direction)
        self._shift_time()

    def _emit_corridor_prep_relation(self, direction: Literal["V", "H"]) -> None:
        """Declare the certified inner-center PREP null relation."""
        corridor = self.geometry.vertical if direction == "V" else self.geometry.horizontal
        records = tuple(self.records.last(corridor.centers[index]) for index in (1, 2))
        parity = self._xor(self._record_parity(record) for record in records)
        if not parity:
            raise AssertionError("empty corridor PREP relation")
        ordered = sorted(
            (self.records.record(name) for name in parity),
            key=lambda record: record.absolute_index,
        )
        (x1, y1), (x2, y2) = (self.geometry.qubit_to_coordinate[record.qubit] for record in records)
        midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
        self.source.append(
            "DETECTOR",
            tuple(self.records.rec_target(record) for record in ordered),
            (*midpoint, 0, TAG_CORRIDOR_PREP_RELATION),
        )
        self.detector_count += 1
        self._register_forward_relation(parity)

    def _emit_verify_detectors(
        self,
        direction: Literal["V", "H"],
        module_name: str,
        round_index: int,
        verify_targets: tuple[int, ...],
        active_patch: tuple[int, ...],
    ) -> None:
        anchors = self._verify_anchor.setdefault(module_name, {})
        path_rows: list[tuple[DetectorCoordinate, frozenset[str]]] = []
        for qubit in verify_targets:
            current = self._record_parity(self.records.last(qubit))
            if round_index == 0:
                anchors[qubit] = current
                atomic = current
            else:
                atomic = current ^ anchors[qubit]
            x, y = self.geometry.qubit_to_coordinate[qubit]
            path_rows.append(((x, y, self.time), atomic))
        # A basis row can use another atomic PATH parity from the same completed
        # M/R handoff even if that detector prints later. Register the complete
        # layer first; all of its measurements are already available.
        for coordinate, atomic in path_rows:
            if coordinate in self._basis_coordinates:
                if coordinate in self._atomic_detectors:
                    raise AssertionError(f"duplicate detector-basis coordinate {coordinate}")
                self._atomic_detectors[coordinate] = atomic
        for coordinate, atomic in path_rows:
            x, y = coordinate[:2]
            self._emit_named_detector(
                coordinate=coordinate,
                coordinate_arguments=(x, y, 0),
                atomic_parity=atomic,
            )

        terminal = round_index + 1 == self.rv
        if terminal:
            corridor = self.geometry.vertical if direction == "V" else self.geometry.horizontal
            current_coupling = {
                q: self._record_parity(self.records.last(q)) for q in corridor.flanks
            }
            previous = self._previous_coupling[direction]
            projective = (not _is_x_memory(self.memory) and direction == "V" and not previous) or (
                _is_x_memory(self.memory) and direction == "H" and not previous
            )
            if not projective:
                atomic = self._xor(current_coupling.values())
                if previous:
                    atomic ^= self._xor(previous.values())
                center = 2 * self.geometry.d + 1
                tag = TAG_VERTICAL_OUTER_PARITY if direction == "V" else TAG_HORIZONTAL_OUTER_PARITY
                self._emit_named_detector(
                    coordinate=(center, center, self.time, tag),
                    coordinate_arguments=(center, center, 0, tag),
                    atomic_parity=atomic,
                )
            self._previous_coupling[direction] = current_coupling

        for qubit in active_patch:
            current = self._record_parity(self.records.last(qubit))
            atomic = current ^ self._patch_reference[qubit]
            x, y = self.geometry.qubit_to_coordinate[qubit]
            self._emit_named_detector(
                coordinate=(x, y, self.time),
                coordinate_arguments=(x, y, 0),
                atomic_parity=atomic,
            )
            self._patch_reference[qubit] = current
        if terminal:
            self._last_module = direction
        self._shift_time()

    def _emit_tail_detectors_and_observables(self) -> None:
        compatible: Basis = "X" if _is_x_memory(self.memory) else "Z"
        for patch in self.geometry.patches.values():
            coordinates = patch.x_ancillas if compatible == "X" else patch.z_ancillas
            for coordinate in coordinates:
                ancilla = self.geometry.q(coordinate)
                atomic = self._patch_reference[ancilla] ^ self._xor(
                    self._record_parity(self.records.last(q))
                    for q in self.geometry.ancilla_data_neighbors[ancilla]
                )
                x, y = coordinate
                self._emit_named_detector(
                    coordinate=(x, y, self.time + 1),
                    coordinate_arguments=(x, y, 1),
                    atomic_parity=atomic,
                )

        center = 2 * self.geometry.d + 1
        if not _is_x_memory(self.memory):
            coupling = self._horizontal_coupling
            corridor_parity = self._xor(self._previous_coupling["H"].values())
            data_qubits = (data for data, _flank in coupling)
        else:
            coupling = self._vertical_coupling
            corridor_parity = self._xor(self._previous_coupling["V"].values())
            data_qubits = (data for _flank, data in coupling)
        atomic = corridor_parity ^ self._xor(
            self._record_parity(self.records.last(q)) for q in data_qubits
        )
        self._emit_named_detector(
            coordinate=(center, center, self.time + 1, TAG_GLOBAL_CLOSURE_PARITY),
            coordinate_arguments=(center, center, 1, TAG_GLOBAL_CLOSURE_PARITY),
            atomic_parity=atomic,
        )

        self._emit_observables()

    def _data_line(
        self,
        patch_id: int,
        *,
        row: Literal["top", "bottom"] | None = None,
        column: Literal["left", "right"] | None = None,
    ) -> tuple[int, ...]:
        coordinates = self.geometry.patches[patch_id].data
        if row is not None:
            selected = (
                min(q[1] for q in coordinates) if row == "top" else max(q[1] for q in coordinates)
            )
            return tuple(self.geometry.q(q) for q in coordinates if q[1] == selected)
        if column is not None:
            selected = (
                min(q[0] for q in coordinates)
                if column == "left"
                else max(q[0] for q in coordinates)
            )
            return tuple(self.geometry.q(q) for q in coordinates if q[0] == selected)
        raise ValueError("choose exactly one row or column")

    def _emit_observable_parity(self, qubits: Sequence[int], observable_id: int) -> None:
        parity = self._canonicalize_frame(
            self._xor(self._record_parity(self.records.last(qubit)) for qubit in qubits)
        )
        ordered = sorted(
            (self.records.record(name) for name in parity),
            key=lambda record: record.absolute_index,
        )
        self.source.append(
            "OBSERVABLE_INCLUDE",
            tuple(self.records.rec_target(record) for record in ordered),
            observable_id,
        )

    def _emit_observables(self) -> None:
        if not _is_x_memory(self.memory):
            first = (*self._data_line(3, row="top"), *self._data_line(4, row="top"))
            second = (*self._data_line(2, row="bottom"), *self._data_line(4, row="top"))
        else:
            first = (*self._data_line(1, column="right"), *self._data_line(3, column="right"))
            second = (*self._data_line(1, column="right"), *self._data_line(2, column="left"))
        self._emit_observable_parity(first, 0)
        self._emit_observable_parity(second, 1)


# =============================================================================
# Public API
# =============================================================================


def _emit_pair(
    writer: _ChronologicalWriter,
    pair_index: int,
    *,
    final: bool = False,
) -> None:
    """Emit one chronological opposite/same-corridor pair."""
    if _uses_zz_flow(writer.memory):
        writer.emit_horizontal_module(f"H{pair_index}")
        writer.emit_vertical_module(f"V{pair_index + 1}", final=final)
    else:
        writer.emit_vertical_module(f"V{pair_index}")
        writer.emit_horizontal_module(f"H{pair_index + 1}", final=final)


def _validate_parameters(d: int, rv: int, ro: int, memory: str) -> tuple[int, int, int, Memory]:
    """Reject fractional, boolean and string dimensions instead of truncating."""
    for name, value, minimum in (("d", d, 2), ("rv", rv, 1), ("ro", ro, 1)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer, got {value!r}")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    if not isinstance(memory, str):
        raise TypeError("memory must be a string")
    memory = memory.upper().replace("_", "-")
    if memory not in ("ZZ", "XX"):
        raise ValueError("memory must be 'ZZ' or 'XX'")
    return int(d), int(rv), int(ro), memory


def _assert_annotation_records_semantic(circuit: stim.Circuit, records: _RecordBook) -> None:
    """Ensure every annotation uses named records; logicals use final data only."""
    semantic = {r.absolute_index for r in records._records.values()}
    final_data = {
        r.absolute_index for name, r in records._records.items() if name.startswith("data.q")
    }
    count = 0
    observables = []
    for instruction in circuit:
        if instruction.name == "M":
            count += len(instruction.targets_copy())
        elif instruction.name in {"DETECTOR", "OBSERVABLE_INCLUDE"}:
            for target in instruction.targets_copy():
                if not target.is_measurement_record_target:
                    raise AssertionError("annotation contains a non-record target")
                index = count + target.value
                if index not in semantic or not 0 <= index < count:
                    raise AssertionError(
                        f"{instruction.name} uses invalid/anonymous record {index}"
                    )
                if instruction.name == "OBSERVABLE_INCLUDE" and index not in final_data:
                    raise AssertionError("logical observable contains nonfinal-data history")
            if instruction.name == "OBSERVABLE_INCLUDE":
                observables.append(int(instruction.gate_args_copy()[0]))
    if count != records.measurement_count or count != circuit.num_measurements:
        raise AssertionError("physical/semantic measurement counts disagree")
    if observables != [0, 1]:
        raise AssertionError("expected exactly two logical observable declarations")


def _build_writer(
    d: int, rv: int, ro: int, memory: str, writer_type: type[_ChronologicalWriter]
) -> _ChronologicalWriter:
    d, rv, ro, memory = _validate_parameters(d, rv, ro, memory)
    writer = writer_type(d=d, rv=rv, memory=memory)
    writer.emit_prep()
    if _uses_zz_flow(memory):
        writer.emit_vertical_module("V0")
    else:
        writer.emit_horizontal_module("H0")
    for index in range(ro):
        _emit_pair(writer, index, final=index == ro - 1)
    writer.emit_data_readout()
    if set(writer._detector_basis_rows) - set(writer._atomic_detectors):
        raise AssertionError("detector-basis row has no physical owner")
    return writer


def _build_circuit(
    d: int, rv: int, ro: int, memory: str, writer_type: type[_ChronologicalWriter]
) -> stim.Circuit:
    writer = _build_writer(d, rv, ro, memory, writer_type)
    circuit = writer.source.parse()
    _assert_annotation_records_semantic(circuit, writer.records)
    return circuit


def build_circuit(t: int, rv: int, ro: int = 1, *, memory: str = "ZZ") -> stim.Circuit:
    """Noiseless memory circuit of the stitched code with both logical observables.

    t >= 2 is the patch distance (code distance 2t), rv >= 1 the number of GHZ
    verification rounds per corridor, ro >= 1 the number of outer-code rounds.
    memory selects the basis in which both logical qubits are prepared and
    measured; the two outer-code generators are measured in opposite orders
    for "ZZ" and "XX".  Noise and compilation to native gates are external.
    """
    return _build_circuit(t, rv, ro, memory, _ChronologicalWriter)


__all__ = ["build_circuit"]
