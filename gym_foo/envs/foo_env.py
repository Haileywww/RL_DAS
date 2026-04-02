import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import gymnasium as gym
from gymnasium.spaces.box import Box
from scipy.spatial import Voronoi
from shapely.geometry import Point, Polygon
import csv
from periodic_voronoi.tile import *
import heapq
import ast
import random
import copy


class FooEnv(gym.Env):
    """
    Custom Gymnasium environment for Dynamic Airspace Sectorisation (DAS) using
    Reinforcement Learning.

    The environment models UK airspace (lat 48–62, lon -7–5) as a set of Voronoi
    sectors whose generator points (centroids) the agent can incrementally adjust.
    Each episode consists of `max_step` 4-hour time windows covering one day of
    traffic. At each step the agent shifts the centroids, the environment rebuilds
    the Voronoi diagram, and three reward signals are evaluated:
      - Sector-crossing count  (workload proxy)
      - Conflict-border proximity (safety proxy)
      - Temporal dis-similarity  (stability proxy)
    """

    def __init__(self, similarity_sample=10000):
        """
        Parameters
        ----------
        similarity_sample : int or None
            Number of flights to randomly sample when computing the similarity
            reward.  Use a positive integer (e.g. 10, 100) for faster
            approximate computation, or ``None`` to check all flights.
        """
        super(FooEnv, self).__init__()

        # Observation space: 19 seed points tiled across 9 surrounding cells
        # gives 19*9 = 171 rows; each row stores (lat, lon, alt_split).
        # Altitude split values are normalised: FL200 -> 4.0, FL350 -> 7.0.
        obs_low = np.full((19 * 9, 3), [48, -7, 200 / 50])
        obs_high = np.full((19 * 9, 3), [62, 5, 350 / 50])

        # Action space: bounded to zero so the pre-trained policy's raw outputs
        # are accepted without clipping while still satisfying the Gymnasium API.
        act_low = np.full((19 * 9, 3), [-1, -1, -1])
        act_high = np.full((19 * 9, 3), [1, 1, 1])

        self.obs = self.load_initial_state()
        self.flight_data = self.load_flight_data()
        self.conflict_data = self.load_conflict_data()
        # Number of flights sampled per step for similarity; None = all flights
        self.similarity_sample = similarity_sample
        # Separate RNG for similarity sampling — isolated from the seed set in
        # load_flight_data so the sample changes every step and every run.
        self._similarity_rng = random.Random()

        self.observation_space = Box(low=obs_low, high=obs_high, shape=(19 * 9, 3), dtype=np.float32)
        self.action_space = Box(low=act_low, high=act_high, shape=(19 * 9, 3), dtype=np.float32)

        self.current_step = 0
        self.max_step = 6 * 7  # number of 4-hour intervals per episode in 1 week
        self.total_rewards = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def step(self, action):
        """
        Advance the environment by one 4-hour interval.

        The action is added directly to the observation (centroid coordinates),
        then the Voronoi diagram is rebuilt and all three reward components are
        computed and combined into a scalar reward:

            reward = -0.3 * (crossing / 10000)
                     -0.4 * (conflict  / 10000)
                     -0.3 * dis_similarity
        """
        self.current_step += 1
        self.obs += action

        vor_dict, poly_dict, vertices = self.get_vor_vertices()

        crossing_reward, sector_waypoints = self.calc_crossing_waypoints(poly_dict, self.current_step)
        similarity_reward = self.calc_similarity(poly_dict, self.current_step)
        conflict_reward = self.calc_conflict(poly_dict, self.conflict_data, self.current_step)

        rewards = round(
            -0.3 * crossing_reward / 10000
            - 0.4 * conflict_reward / 10000
            - 0.3 * similarity_reward,
            3,
        )
        self.total_rewards += rewards

        # Cache current sectorisation for similarity comparison in the next step
        self.pre_vor_dict = vor_dict
        self.pre_poly_dict = poly_dict

        info = {
            'step': self.current_step,
            'conflict': conflict_reward,
            'crossing': crossing_reward,
            'dis-similarity': similarity_reward,
            'centroids': self.centroids_arr,
            'polygons': poly_dict,
        }

        terminated = False
        truncated = (self.current_step >= self.max_step)
        if truncated:
            print('total_reward', self.total_rewards)

        return np.float32(self.obs), rewards, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the environment to the fixed initial sectorisation."""
        self.obs = self.load_initial_state()
        self.current_step = 0
        self.total_rewards = 0
        return np.float32(self.obs), {}

    def close(self):
        print('environment closed')

    # ------------------------------------------------------------------
    # State initialisation
    # ------------------------------------------------------------------

    def load_initial_state(self):
        """
        Build the initial observation array from 19 hand-picked UK sector seed
        points.  The seeds are tiled into a 3×3 grid of copies (9 tiles × 19
        points = 171 rows) so that the periodic Voronoi diagram has no boundary
        artefacts.  A fixed altitude split of FL275 (5.5) is appended as the
        third column.
        """
        # 19 seed centroids (lat, lon) covering original UK airspace sectorization
        cleaned_states = np.array([
            [52.04, 0.26], [51.64, 2.5],  [52.1, -0.97], [53.2, -5.5],
            [50.4,  0.77], [51.17, -2.0], [51.48, -3.76], [49.48, -4.92],
            [58.03, -6.49],[55.38, 0.37], [56.27, -3.49], [59.28, -4.5],
            [55.85, 3.5],  [54.01, -2.46],[55.94, -5.38], [54.65, 4.33],
            [61.0, -5.5],  [50.5, -0.54], [55.0, -1.5],
        ])

        initial_voronoi_array = tile_points(cleaned_states, 19)

        # Append a uniform initial altitude split (FL275 = 5.5 in 50-ft units)
        initial_state_array = np.column_stack(
            (initial_voronoi_array, np.full((initial_voronoi_array.shape[0],), 5.5))
        )
        return initial_state_array

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_flight_data(self, scale=1, seed=42):
        """
        Load flight data and apply uniform thinning (scale < 1) or boosting
        (scale > 1).

        Parameters
        ----------
        scale : float
            Demand scaling factor (e.g. 0.8 for −20 %, 1.2 for +20 %, 1.0 for
            baseline).
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        flight_data : dict
            Mapping ``{icao24: [lat_list, lon_list, alt_list, time_list]}``
            after the requested scaling has been applied.
        """
        rng = random.Random(seed)

        # Load the rows
        flight_data = {}
        with open('train_flight_data.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
            rows = list(csvreader)
            count = len(rows) // 5  # your original sampling
            rows = rows[: count]

            for row in rows:
                icao24 = row[0]
                flight_data[icao24] = [
                    ast.literal_eval(row[1]),  # lat_list
                    ast.literal_eval(row[2]),  # lon_list
                    ast.literal_eval(row[3]),  # alt_list
                    ast.literal_eval(row[4]),  # time_list
                ]

        icao_keys = list(flight_data.keys())

        # Thinning: keep each flight with probability = scale
        if scale < 1.0:
            kept_keys = [k for k in icao_keys if rng.random() < scale]
            flight_data = {k: flight_data[k] for k in kept_keys}

        # Boosting: duplicate randomly selected flights with a small time jitter
        elif scale > 1.0:
            n_extra = int(round((scale - 1.0) * len(icao_keys)))
            for key in rng.choices(icao_keys, k=n_extra):
                new_key = key + "_dup" + str(rng.randint(1000, 9999))
                lat_list, lon_list, alt_list, time_list = copy.deepcopy(flight_data[key])

                # Convert timestamps to float, dropping any non-numeric entries
                # (e.g. stray header strings such as 'time       ')
                valid_times = []
                for t in time_list:
                    try:
                        valid_times.append(float(t))
                    except (ValueError, TypeError):
                        continue
                time_list = valid_times

                # Apply a small random time shift (±30 seconds)
                jitter = rng.randint(-30, 30)
                time_list = [t + jitter for t in time_list]

                flight_data[new_key] = [lat_list, lon_list, alt_list, time_list]

        return flight_data

    def load_conflict_data(self):
        """
        Load pre-computed conflict events from CSV.

        Each row represents one conflict event stored as:
            [[conf_lat, conf_lon], conf_alt, conf_time]
        """
        conflict_data = []
        with open('test_conflict_data.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
            for row in csvreader:
                conflict_data.append([
                    [ast.literal_eval(row[0]), ast.literal_eval(row[1])],  # [lat, lon]
                    ast.literal_eval(row[2]),                               # altitude (FL)
                    ast.literal_eval(row[3]),                               # UNIX timestamp
                ])
        return conflict_data

    # ------------------------------------------------------------------
    # Voronoi construction
    # ------------------------------------------------------------------

    def get_vor_vertices(self):
        """
        Build a clipped, periodic Voronoi diagram for the current centroids.

        The first N rows of ``self.obs`` are used as Voronoi generators.
        Points are tiled across 9 cells to avoid boundary effects (periodic
        Voronoi).  The resulting polygons are clipped to the UK bounding box
        (lat 48–62, lon -7–5).  If more than 19 polygons remain after clipping,
        the smallest excess ones are merged into their smallest neighbour until
        exactly 19 sectors remain.

        Returns
        -------
        vor_dict  : dict  {centroid_tuple: raw_vertex_list}
        poly_dict : dict  {centroid_tuple: shapely.Polygon}
        vertices  : ndarray  raw Voronoi vertices
        """
        N = 19  # number of active sector generators used from self.obs
        box = Polygon([(48, -7), (48, 5), (62, 5), (62, -7), (48, -7)])

        self.centroids = []
        self.splits = []
        for n in range(N):
            lat, lon, alt = self.obs[n]
            self.centroids.append([lat, lon])
            self.splits.append(alt)

        self.centroids_arr = np.array(self.centroids)
        self.point_tile = tile_points(self.centroids_arr, N)

        vor = Voronoi(self.point_tile)
        vertices = vor.vertices
        regions = vor.regions
        point_region = vor.point_region

        # Collect closed (finite) Voronoi regions
        vor_dict = {}
        poly_dict = {}
        area_dict = {}
        final_vertices = []
        for i in range(len(point_region)):
            region_idx = point_region[i]
            vertice_idx = regions[region_idx]
            if -1 not in vertice_idx:
                final_vertices.append([list(vertices[v]) for v in vertice_idx])

        for point, vertice in zip(vor.points, final_vertices):
            clipped_poly = Polygon(vertice).intersection(box)
            if clipped_poly:
                vor_dict[tuple(point)] = vertice
                poly_dict[tuple(point)] = clipped_poly
                area_dict[tuple(point)] = clipped_poly.area

        # Merge surplus small polygons until exactly 19 sectors remain
        if len(area_dict) != 19:
            smallest_areas = heapq.nsmallest(len(area_dict) - 19, area_dict.values())
            smallest_polys = {
                key: Polygon(poly_dict[key])
                for key, value in area_dict.items()
                if value in smallest_areas
            }
            poly_dict = {k: v for k, v in poly_dict.items() if k not in smallest_polys}

            combined_polys = {}
            for key1, smallest_poly in smallest_polys.items():
                # Find all polygons that intersect (or touch) this small polygon
                intersect_polys = [
                    (k, p) for k, p in poly_dict.items()
                    if smallest_poly != p and smallest_poly.intersection(p)
                ]
                if not intersect_polys:
                    intersect_polys = [
                        (k, p) for k, p in poly_dict.items()
                        if smallest_poly != p and smallest_poly.touches(p)
                    ]
                if intersect_polys:
                    key2, smallest_intersect_poly = min(intersect_polys, key=lambda x: x[1].area)
                    combined_polys[key2] = smallest_poly.union(smallest_intersect_poly)

            for key2, combined_poly in combined_polys.items():
                poly_dict[key2] = combined_poly

        return vor_dict, poly_dict, vertices

    # ------------------------------------------------------------------
    # Reward components
    # ------------------------------------------------------------------

    def calc_crossing_waypoints(self, poly_dict, num_step):
        """
        Count the number of unique (flight, sector) crossings in the current
        4-hour window.

        Where a sector has an altitude split (4 < alt_split < 7), the sector is
        treated as two vertical sub-sectors ('up' / 'down') and crossings are
        counted separately for each.

        Parameters
        ----------
        poly_dict : dict
            Current sectorisation {centroid: shapely.Polygon}.
        num_step : int
            Current episode step (1-indexed), used to derive the time window.

        Returns
        -------
        num_crossings : int
            Total unique sector-crossing count.
        sectors_wps : dict
            Mapping {sector_key: waypoint_count} for each sector (or sub-sector).
        """
        start_time = 1673913600 + (num_step - 1) * 4 * 3600
        end_time = start_time + 4 * 3600
        num_crossings = 0
        sectors_wps = {}

        for flight_id, data in self.flight_data.items():
            x, y, z = [], [], []
            for lat, lon, alt, t in zip(data[0], data[1], data[2], data[3]):
                # Skip stray header strings that survive CSV parsing
                if t != 'time       ':
                    if start_time <= int(t) <= end_time:
                        x.append(lat)
                        y.append(lon)
                        z.append(alt)

            # Shapely uses (x, y) = (lon, lat) convention
            trajectory_points = [Point(lon, lat) for lat, lon in zip(y, x)]
            sectors_crossed = set()

            for centroid, polygon in poly_dict.items():
                for idx, point in enumerate(trajectory_points):
                    if polygon.contains(point):
                        c_idx = np.where(self.point_tile == list(centroid))
                        p_alt = (float(z[idx]) * 3.28) // 100  # metres → flight level

                        if 4 < self.obs[c_idx][2] < 7:
                            # Sector has a vertical split
                            sub = 'up' if p_alt >= (round(self.obs[c_idx][2]) * 50) else 'down'
                            sectors_crossed.add((centroid, sub))
                            sectors_wps[(centroid, sub)] = sectors_wps.get((centroid, sub), 0) + 1
                        else:
                            sectors_crossed.add(centroid)
                            sectors_wps[centroid] = sectors_wps.get(centroid, 0) + 1

            num_crossings += len(sectors_crossed)

        return num_crossings, sectors_wps

    def calc_sd_density(self, sector_density):
        """
        Compute the sample standard deviation of normalised sector densities.

        Density values are L2-normalised before computing the standard deviation
        so that the metric is scale-invariant.  Returns 0 when no sectors are
        provided.

        Note: this metric is computed but not currently included in the reward.
        """
        if not sector_density:
            return 0

        sum_of_squares = sum(d ** 2 for d in sector_density.values())
        if sum_of_squares == 0:
            return 0

        norm_density = [np.sqrt((d ** 2) / sum_of_squares) for d in sector_density.values()]

        mean = sum(norm_density) / len(norm_density)
        variance_sum = sum((n - mean) ** 2 for n in norm_density)
        divisor = len(norm_density) - 1 if len(norm_density) > 1 else 1
        return round(np.sqrt(variance_sum / divisor), 2)

    def calc_similarity(self, poly_dict, num_step):
        """
        Measure temporal dis-similarity between the current and previous
        sectorisation.

        For each aircraft position in the current 4-hour time window, checks
        whether the aircraft would belong to a different sector if a
        re-sectorisation occurred (i.e. current poly_dict vs previous
        self.pre_poly_dict).  The fraction of aircraft positions that change
        sector is returned as the dissimilarity ratio.

        Returns 0 at the first step (no previous sectorisation to compare
        against).
        """
        dissimilarity = 0

        if num_step == 1:
            # No previous sectorisation to compare against
            return 0.0

        # --- OLD grid-based approach (commented out) ---
        # resolution = 10
        # lats = np.linspace(48, 62, num=resolution)
        # lons = np.linspace(-7, 5, num=resolution)
        # vor_list = list(poly_dict.keys())
        # pre_poly_list = list(self.pre_poly_dict.values())
        #
        # for centroid, polygon in poly_dict.items():
        #     p_idx = vor_list.index(centroid)
        #     for lat in lats:
        #         for lon in lons:
        #             if polygon.contains(Point(lat, lon)):
        #                 if len(pre_poly_list) == len(vor_list):
        #                     if not Polygon(pre_poly_list[p_idx]).contains(Point(lat, lon)):
        #                         dissimilarity += 1
        #                 else:
        #                     # Sector count changed — treat as maximum dissimilarity
        #                     dissimilarity = 50
        #
        # return dissimilarity / (resolution ** 2)
        # --- END OLD approach ---

        # New approach: use actual aircraft positions in the current time window.
        # For each aircraft waypoint, find which sector it belongs to under the
        # current and previous sectorisations.  If re-sectorisation would move
        # an aircraft to a different sector, count it as dissimilar.
        start_time = 1673913600 + (num_step - 1) * 4 * 3600
        end_time = start_time + 4 * 3600

        all_flight_ids = list(self.flight_data.keys())
        if self.similarity_sample is not None and self.similarity_sample < len(all_flight_ids):
            sampled_ids = self._similarity_rng.sample(all_flight_ids, self.similarity_sample)
        else:
            sampled_ids = all_flight_ids

        # Use ordered lists so sectors can be matched by index across timesteps.
        # Centroid keys cannot be compared directly — they change every step as
        # centroids move — so index-based matching (same order, same sector) is
        # used instead, mirroring the original grid-based approach.
        poly_list     = list(poly_dict.values())
        pre_poly_list = list(self.pre_poly_dict.values())

        # Count flights (not waypoints): a flight is "changed" if at least one
        # of its waypoints would land in a different sector after re-sectorisation.
        total_flights = 0  # sampled flights that have at least one waypoint in the window

        for flight_id in sampled_ids:
            data = self.flight_data[flight_id]
            flight_changed = False
            has_waypoint_in_window = False

            for lat, lon, alt, t in zip(data[0], data[1], data[2], data[3]):
                # Skip stray header strings that survive CSV parsing
                if t == 'time       ':
                    continue
                if not (start_time <= int(t) <= end_time):
                    continue

                has_waypoint_in_window = True

                # Polygon coordinate system is (lat, lon) — consistent with
                # the Voronoi generators stored as [lat, lon] in self.centroids.
                point = Point(lat, lon)

                # Find the index of the sector containing this waypoint in the
                # current sectorisation.
                curr_idx = None
                for i, polygon in enumerate(poly_list):
                    if polygon.contains(point):
                        curr_idx = i
                        break

                if len(pre_poly_list) == len(poly_list):
                    if curr_idx is not None:
                        # Waypoint changes sector if the same-indexed polygon in
                        # the previous sectorisation does NOT contain it.
                        if not pre_poly_list[curr_idx].contains(point):
                            flight_changed = True
                            break  # one changed waypoint is enough
                    else:
                        # Waypoint outside all current sectors — treat as changed
                        flight_changed = True
                        break
                else:
                    # Sector count changed — treat as maximum dissimilarity
                    flight_changed = True
                    break

            if has_waypoint_in_window:
                total_flights += 1
                if flight_changed:
                    dissimilarity += 1

        if total_flights == 0:
            return 0.0

        return dissimilarity / total_flights

    def calc_conflict(self, poly_dict, conflict_data, num_step):
        """
        Count conflict events that occur close to (or on) sector boundaries.

        A conflict is flagged as a *border conflict* if:
        - The conflict location is within `threshold_distance` nautical miles of
          a sector boundary (distance > 0), **or**
        - The conflict is exactly on the boundary (distance == 0) and the sector
          has an altitude split within ±10 FL of the conflict altitude.

        Parameters
        ----------
        poly_dict      : dict   Current sectorisation.
        conflict_data  : list   Pre-loaded conflict events (from load_conflict_data).
        num_step       : int    Current episode step.

        Returns
        -------
        num_border_conflict : int
        """
        start_time = 1673913600 + (num_step - 1) * 4 * 3600
        end_time = start_time + 4 * 3600

        sector_borders = list(poly_dict.items())
        threshold_distance = 5  # nautical miles (approximate, mapped to degrees)
        obs_set = {(obs[0], obs[1]) for obs in self.obs}
        num_border_conflict = 0

        for conflict in conflict_data:
            if start_time <= conflict[2] < end_time:
                for sector_centroid, sector_border in sector_borders:
                    distance = Point(conflict[0]).distance(sector_border)
                    if 0 < distance <= (threshold_distance / 60):
                        num_border_conflict += 1
                    elif distance == 0:
                        # Conflict is on the boundary — check vertical separation
                        if (sector_centroid[0], sector_centroid[1]) in obs_set:
                            for i in range(len(self.obs)):
                                if 4 < self.obs[i][2] < 7:
                                    if abs(round(self.obs[i][2]) * 50 - conflict[1]) <= 10:
                                        num_border_conflict += 1

        return num_border_conflict
