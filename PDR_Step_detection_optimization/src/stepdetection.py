import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib._multiprocessing_helpers import mp
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import os
from typing import List, Dict, Tuple, Optional
import math

from scipy.optimize import minimize


class StepDetectionAlgorithm:
    """
    Step Detection Algorithm based on the paper:
    "Step Detection Algorithm For Accurate Distance Estimation Using Dynamic Step Length"

    This implementation follows the two-phase approach:
    1. Peak detection phase
    2. Dynamic step length estimation phase
    """

    def __init__(self,
                 step_threshold: float = 0.5,
                 min_peak_distance: int = 50,
                 window_size: int = 10,
                 map_width_cm: float = 2643.49,
                 map_height_cm: float = 3264.33,
                 grid_width: int = 928,
                 grid_height: int = 1167):
        """
        Initialize the step detection algorithm.

        Args:
            step_threshold: Threshold for step detection
            min_peak_distance: Minimum distance between peaks (in samples)
            window_size: Size of the sliding window for smoothing
            map_width_cm: Real-world map width in cm
            map_height_cm: Real-world map height in cm
            grid_width: Grid width in pixels
            grid_height: Grid height in pixels
        """
        self.step_threshold = step_threshold
        self.min_peak_distance = min_peak_distance
        self.window_size = window_size
        self.map_width_cm = map_width_cm
        self.map_height_cm = map_height_cm
        self.grid_width = grid_width
        self.grid_height = grid_height

    def convert_grid_to_real_world(self, grid_x: float, grid_y: float) -> Tuple[float, float]:
        """
        Convert grid coordinates to real-world coordinates (cm).

        Args:
            grid_x: X coordinate in grid
            grid_y: Y coordinate in grid

        Returns:
            Tuple of (real_world_x, real_world_y) in cm
        """
        real_world_x = (grid_x / self.grid_width) * self.map_width_cm
        real_world_y = (grid_y / self.grid_height) * self.map_height_cm
        return real_world_x, real_world_y

    def calculate_magnitude(self, x: float, y: float, z: float) -> float:
        """
        Calculate magnitude from accelerometer values.

        Args:
            x, y, z: Accelerometer values

        Returns:
            Magnitude value
        """
        return math.sqrt(x ** 2 + y ** 2 + z ** 2)

    def remove_gravity(self, magnitudes: List[float]) -> List[float]:
        """
        Remove gravity component from magnitude values.

        Args:
            magnitudes: List of magnitude values

        Returns:
            List of net magnitude values (gravity removed)
        """
        avg_mag = np.mean(magnitudes)
        net_magnitudes = [mag - avg_mag for mag in magnitudes]
        return net_magnitudes

    def apply_filters(self, data: List[float]) -> List[float]:
        """
        Apply filtering to remove noise and outliers.

        Args:
            data: Raw sensor data

        Returns:
            Filtered data
        """
        # Convert to numpy array for processing
        data_array = np.array(data)

        # Apply Gaussian smoothing (similar to Kalman filter effect)
        smoothed_data = gaussian_filter1d(data_array, sigma=1.0)

        # Apply high-pass filter to remove low-frequency noise
        b, a = signal.butter(2, 0.1, btype='high', analog=False)
        filtered_data = signal.filtfilt(b, a, smoothed_data)

        return filtered_data.tolist()

    def detect_peaks_enhanced(self, magnitudes: List[float], timestamps: List[int]) -> List[Dict]:
        """
        Enhanced peak detection algorithm based on the paper's approach.

        Args:
            magnitudes: List of magnitude values
            timestamps: List of corresponding timestamps

        Returns:
            List of detected peaks with metadata
        """
        peaks = []
        i = 0

        while i < len(magnitudes) - 2:
            # Check for three consecutive increasing values (Start Vector)
            if (magnitudes[i] < magnitudes[i + 1] < magnitudes[i + 2]):
                start_idx = i
                start_value = magnitudes[i]

                # Find values above threshold (Peak Vector)
                peak_vector = []
                peak_indices = []
                j = i

                while j < len(magnitudes) and magnitudes[j] > self.step_threshold:
                    peak_vector.append(magnitudes[j])
                    peak_indices.append(j)
                    j += 1

                if peak_vector:
                    # Find maximum value in peak vector (real peak)
                    max_idx = peak_indices[np.argmax(peak_vector)]
                    max_value = max(peak_vector)

                    # Find end point (two consecutive increasing values)
                    end_idx = max_idx
                    while end_idx < len(magnitudes) - 1:
                        if magnitudes[end_idx] < magnitudes[end_idx + 1]:
                            break
                        end_idx += 1

                    # Calculate step length (time duration)
                    step_length = end_idx - start_idx

                    # Only accept peaks with sufficient distance from previous peak
                    if not peaks or (max_idx - peaks[-1]['peak_index']) >= self.min_peak_distance:
                        peak_info = {
                            'peak_index': max_idx,
                            'peak_value': max_value,
                            'start_index': start_idx,
                            'end_index': end_idx,
                            'start_value': start_value,
                            'end_value': magnitudes[end_idx] if end_idx < len(magnitudes) else magnitudes[-1],
                            'step_length_samples': step_length,
                            'timestamp': timestamps[max_idx] if max_idx < len(timestamps) else timestamps[-1],
                            'peak_vector': peak_vector
                        }
                        peaks.append(peak_info)

                    i = end_idx + 1
                else:
                    i += 1
            else:
                i += 1

        return peaks

    def categorize_step_length(self, step_lengths: List[int]) -> List[Dict]:
        """
        Categorize steps into short, medium, and long based on average.

        Args:
            step_lengths: List of step lengths in samples

        Returns:
            List of categorized steps with weights
        """
        if not step_lengths:
            return []

        avg_step_length = np.mean(step_lengths)

        categorized_steps = []
        for step_length in step_lengths:
            if step_length < avg_step_length:
                category = 'short'
                weight = 0.5
            elif step_length == avg_step_length:
                category = 'medium'
                weight = 1.0
            else:
                category = 'long'
                weight = 1.5

            categorized_steps.append({
                'step_length': step_length,
                'category': category,
                'weight': weight
            })

        return categorized_steps

    def estimate_distance(self, peaks: List[Dict], categorized_steps: List[Dict]) -> float:
        """
        Estimate total distance using dynamic step length.

        Args:
            peaks: List of detected peaks
            categorized_steps: List of categorized steps with weights

        Returns:
            Estimated distance in cm
        """
        if len(peaks) != len(categorized_steps):
            print(f"Warning: Mismatch between peaks ({len(peaks)}) and categorized steps ({len(categorized_steps)})")
            return 0.0

        total_distance = 0.0

        for i, (peak, step_info) in enumerate(zip(peaks, categorized_steps)):
            # Base step length (can be adjusted based on user characteristics)
            base_step_length_cm = 60.0  # Average step length in cm

            # Apply weight based on step category
            weighted_step_length = base_step_length_cm * step_info['weight']

            total_distance += weighted_step_length

            print(f"Step {i + 1}: Category={step_info['category']}, "
                  f"Weight={step_info['weight']}, "
                  f"Step Length={weighted_step_length:.2f}cm")

        return total_distance

    def process_segment(self, segment_file: str) -> Dict:
        """
        Process a single segment file and detect steps.

        Args:
            segment_file: Path to the segment CSV file

        Returns:
            Dictionary containing step detection results
        """
        print(f"\n=== Processing {segment_file} ===")

        # Load segment data
        df = pd.read_csv(segment_file)

        # Extract accelerometer data
        accel_data = df[df['sensor_type'] == 'TYPE_ACCELEROMETER'].copy()

        if accel_data.empty:
            print("No accelerometer data found in segment")
            return {}

        # Calculate magnitude for each accelerometer reading
        magnitudes = []
        timestamps = []

        for _, row in accel_data.iterrows():
            mag = self.calculate_magnitude(row['x'], row['y'], row['z'])
            magnitudes.append(mag)
            timestamps.append(row['timestamp'])

        print(f"Calculated {len(magnitudes)} magnitude values")

        # Remove gravity component
        net_magnitudes = self.remove_gravity(magnitudes)
        print(f"Removed gravity component")

        # Apply filters
        filtered_magnitudes = self.apply_filters(net_magnitudes)
        print(f"Applied filters")

        # Detect peaks
        peaks = self.detect_peaks_enhanced(filtered_magnitudes, timestamps)
        print(f"Detected {len(peaks)} peaks")

        # Extract step lengths
        step_lengths = [peak['step_length_samples'] for peak in peaks]

        # Categorize steps
        categorized_steps = self.categorize_step_length(step_lengths)

        # Estimate distance
        estimated_distance = self.estimate_distance(peaks, categorized_steps)

        # Calculate real-world distance from grid coordinates
        segment_name = os.path.basename(segment_file)
        grid_coords = self.extract_grid_coordinates(segment_name)

        if grid_coords:
            start_x, start_y = grid_coords['start']
            end_x, end_y = grid_coords['end']

            real_start_x, real_start_y = self.convert_grid_to_real_world(start_x, start_y)
            real_end_x, real_end_y = self.convert_grid_to_real_world(end_x, end_y)

            real_distance = math.sqrt((real_end_x - real_start_x) ** 2 + (real_end_y - real_start_y) ** 2)
        else:
            real_distance = 0.0

        results = {
            'segment_file': segment_file,
            'num_peaks': len(peaks),
            'estimated_distance_cm': estimated_distance,
            'real_distance_cm': real_distance,
            'error_cm': abs(estimated_distance - real_distance),
            'error_percentage': (
                        abs(estimated_distance - real_distance) / real_distance * 100) if real_distance > 0 else 0,
            'peaks': peaks,
            'categorized_steps': categorized_steps,
            'magnitudes': magnitudes,
            'filtered_magnitudes': filtered_magnitudes,
            'timestamps': timestamps
        }

        print(f"Results: {len(peaks)} steps, "
              f"Estimated: {estimated_distance:.2f}cm, "
              f"Real: {real_distance:.2f}cm, "
              f"Error: {results['error_cm']:.2f}cm ({results['error_percentage']:.1f}%)")

        return results

    def extract_grid_coordinates(self, segment_name: str) -> Optional[Dict]:
        """
        Extract grid coordinates from segment filename.

        Args:
            segment_name: Segment filename

        Returns:
            Dictionary with start and end coordinates, or None if parsing fails
        """
        try:
            # Remove .csv extension and split by underscores
            parts = segment_name.replace('.csv', '').split('_')

            # Handle combined segment format: combined_path1_0_2_319_143_to_319_317
            if parts[0] == 'combined':
                if len(parts) >= 7:
                    # Check if there's a source file identifier
                    if len(parts) >= 8 and parts[1] in ['path1', 'path2', 'path3']:
                        # Format: combined_path1_0_2_319_143_to_319_317
                        start_x = float(parts[4])
                        start_y = float(parts[5])
                        end_x = float(parts[7])
                        end_y = float(parts[8])
                    else:
                        # Format: combined_0_2_319_143_to_319_317
                        start_x = float(parts[3])
                        start_y = float(parts[4])
                        end_x = float(parts[6])
                        end_y = float(parts[7])

                    return {
                        'start': (start_x, start_y),
                        'end': (end_x, end_y)
                    }
            else:
                # Handle regular segment format: segment_path1_000_319_143_to_319_158
                if len(parts) >= 6:
                    # Check if there's a source file identifier
                    if len(parts) >= 7 and parts[1] in ['path1', 'path2', 'path3']:
                        # Format: segment_path1_000_319_143_to_319_158
                        start_x = float(parts[3])
                        start_y = float(parts[4])
                        end_x = float(parts[6])
                        end_y = float(parts[7])
                    else:
                        # Format: segment_000_319_143_to_319_158
                        start_x = float(parts[2])
                        start_y = float(parts[3])
                        end_x = float(parts[5])
                        end_y = float(parts[6])

                    return {
                        'start': (start_x, start_y),
                        'end': (end_x, end_y)
                    }
        except (ValueError, IndexError) as e:
            print(f"Error parsing coordinates from filename {segment_name}: {e}")

        return None

    def plot_results(self, results: Dict, save_path: str = None):
        """
        Plot the step detection results.

        Args:
            results: Results dictionary from process_segment
            save_path: Optional path to save the plot
        """
        if not results:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Plot 1: Magnitude values and peaks
        magnitudes = results['magnitudes']
        filtered_magnitudes = results['filtered_magnitudes']
        peaks = results['peaks']

        ax1.plot(magnitudes, label='Raw Magnitude', alpha=0.7)
        ax1.plot(filtered_magnitudes, label='Filtered Magnitude', linewidth=2)

        # Plot peaks
        peak_indices = [peak['peak_index'] for peak in peaks]
        peak_values = [peak['peak_value'] for peak in peaks]
        ax1.scatter(peak_indices, peak_values, color='red', s=50, label='Detected Peaks', zorder=5)

        ax1.axhline(y=self.step_threshold, color='orange', linestyle='--', label='Step Threshold')
        ax1.set_title(f'Step Detection Results - {os.path.basename(results["segment_file"])}')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Magnitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Step length distribution
        if results['categorized_steps']:
            step_lengths = [step['step_length'] for step in results['categorized_steps']]
            categories = [step['category'] for step in results['categorized_steps']]

            colors = {'short': 'red', 'medium': 'blue', 'long': 'green'}
            category_colors = [colors[cat] for cat in categories]

            ax2.bar(range(len(step_lengths)), step_lengths, color=category_colors, alpha=0.7)
            ax2.set_title('Step Length Distribution by Category')
            ax2.set_xlabel('Step Number')
            ax2.set_ylabel('Step Length (samples)')
            ax2.grid(True, alpha=0.3)

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=colors[cat], label=cat.capitalize())
                               for cat in ['short', 'medium', 'long']]
            ax2.legend(handles=legend_elements)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def process_all_segments(self, segments_dir: str, output_dir: str = None) -> List[Dict]:
        """
        Process all segment files in a directory.

        Args:
            segments_dir: Directory containing segment CSV files
            output_dir: Optional directory to save plots

        Returns:
            List of results for all segments
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        all_results = []
        segment_files = [f for f in os.listdir(segments_dir) if f.endswith('.csv')]
        segment_files.sort()

        print(f"Found {len(segment_files)} segment files to process")

        for segment_file in segment_files:
            file_path = os.path.join(segments_dir, segment_file)
            results = self.process_segment(file_path)

            if results:
                all_results.append(results)

                # Save plot if output directory is specified
                if output_dir:
                    plot_filename = f"step_detection_{os.path.splitext(segment_file)[0]}.png"
                    plot_path = os.path.join(output_dir, plot_filename)
                    self.plot_results(results, plot_path)

        # Print summary
        self.print_summary(all_results)

        return all_results

    def print_summary(self, results: List[Dict]):
        """
        Print a summary of all results.

        Args:
            results: List of results from all segments
        """
        if not results:
            return

        print("\n" + "=" * 60)
        print("STEP DETECTION SUMMARY")
        print("=" * 60)

        total_estimated_distance = sum(r['estimated_distance_cm'] for r in results)
        total_real_distance = sum(r['real_distance_cm'] for r in results)
        total_error = sum(r['error_cm'] for r in results)
        total_steps = sum(r['num_peaks'] for r in results)

        avg_error = total_error / len(results) if results else 0
        avg_error_percentage = (total_error / total_real_distance * 100) if total_real_distance > 0 else 0

        print(f"Total Segments Processed: {len(results)}")
        print(f"Total Steps Detected: {total_steps}")
        print(f"Total Estimated Distance: {total_estimated_distance:.2f} cm")
        print(f"Total Real Distance: {total_real_distance:.2f} cm")
        print(f"Total Error: {total_error:.2f} cm")
        print(f"Average Error per Segment: {avg_error:.2f} cm")
        print(f"Average Error Percentage: {avg_error_percentage:.2f}%")

        print("\nSegment Details:")
        print("-" * 60)
        for i, result in enumerate(results):
            segment_name = os.path.basename(result['segment_file'])
            print(f"{i + 1:2d}. {segment_name}")
            print(f"    Steps: {result['num_peaks']:3d}, "
                  f"Est: {result['estimated_distance_cm']:6.2f}cm, "
                  f"Real: {result['real_distance_cm']:6.2f}cm, "
                  f"Error: {result['error_cm']:5.2f}cm ({result['error_percentage']:4.1f}%)")

    def extract_grid_coordinates(self, segment_name: str) -> Optional[Dict]:
        """
        Extract grid coordinates from segment filename.

        Args:
            segment_name: Segment filename

        Returns:
            Dictionary with start and end coordinates, or None if parsing fails
        """
        try:
            # Remove .csv extension and split by underscores
            parts = segment_name.replace('.csv', '').split('_')

            # Handle combined segment format: combined_path1_0_2_319_143_to_319_317
            if parts[0] == 'combined':
                if len(parts) >= 7:
                    # Check if there's a source file identifier
                    if len(parts) >= 8 and parts[1] in ['path1', 'path2', 'path3']:
                        # Format: combined_path1_0_2_319_143_to_319_317
                        start_x = float(parts[4])
                        start_y = float(parts[5])
                        end_x = float(parts[7])
                        end_y = float(parts[8])
                    else:
                        # Format: combined_0_2_319_143_to_319_317
                        start_x = float(parts[3])
                        start_y = float(parts[4])
                        end_x = float(parts[6])
                        end_y = float(parts[7])

                    return {
                        'start': (start_x, start_y),
                        'end': (end_x, end_y)
                    }
            else:
                # Handle regular segment format: segment_path1_000_319_143_to_319_158
                if len(parts) >= 6:
                    # Check if there's a source file identifier
                    if len(parts) >= 7 and parts[1] in ['path1', 'path2', 'path3']:
                        # Format: segment_path1_000_319_143_to_319_158
                        start_x = float(parts[3])
                        start_y = float(parts[4])
                        end_x = float(parts[6])
                        end_y = float(parts[7])
                    else:
                        # Format: segment_000_319_143_to_319_158
                        start_x = float(parts[2])
                        start_y = float(parts[3])
                        end_x = float(parts[5])
                        end_y = float(parts[6])

                    return {
                        'start': (start_x, start_y),
                        'end': (end_x, end_y)
                    }
        except (ValueError, IndexError) as e:
            print(f"Error parsing coordinates from filename {segment_name}: {e}")

        return None

    from scipy.optimize import minimize
    import random
    def optimize_parameters_pso(self, segments_dir: str, n_particles: int = 20, n_iterations: int = 50) -> Dict:
        """
        Optimize parameters using Particle Swarm Optimization.

        Args:
            segments_dir: Directory containing segment CSV files
            n_particles: Number of particles in the swarm
            n_iterations: Number of iterations for optimization

        Returns:
            Dictionary with best parameters and results
        """
        import random

        # Parameter bounds
        bounds = {
            'step_threshold': (0.2, 1.0),
            'min_peak_distance': (15, 80),
            'base_step_length_cm': (40, 80)
        }

        # Initialize particles
        particles = []
        velocities = []
        personal_best = []
        personal_best_fitness = []
        global_best = None
        global_best_fitness = float('inf')

        # Initialize swarm
        for i in range(n_particles):
            # Random initial position
            particle = {
                'step_threshold': random.uniform(bounds['step_threshold'][0], bounds['step_threshold'][1]),
                'min_peak_distance': random.uniform(bounds['min_peak_distance'][0], bounds['min_peak_distance'][1]),
                'base_step_length_cm': random.uniform(bounds['base_step_length_cm'][0],
                                                      bounds['base_step_length_cm'][1])
            }

            # Random initial velocity
            velocity = {
                'step_threshold': random.uniform(-0.1, 0.1),
                'min_peak_distance': random.uniform(-5, 5),
                'base_step_length_cm': random.uniform(-2, 2)
            }

            particles.append(particle)
            velocities.append(velocity)
            personal_best.append(particle.copy())
            personal_best_fitness.append(float('inf'))

        # PSO parameters
        w = 0.7  # Inertia weight
        c1 = 1.5  # Cognitive learning factor
        c2 = 1.5  # Social learning factor

        print(f"Starting PSO optimization with {n_particles} particles for {n_iterations} iterations...")

        for iteration in range(n_iterations):
            if iteration % 10 == 0:
                print(f"Iteration {iteration + 1}/{n_iterations}")

            for i in range(n_particles):
                # Evaluate current particle
                self.step_threshold = particles[i]['step_threshold']
                self.min_peak_distance = int(particles[i]['min_peak_distance'])

                results = self.process_all_segments_optimization(
                    segments_dir,
                    particles[i]['base_step_length_cm']
                )

                if results:
                    fitness = sum(r['error_cm'] for r in results)

                    # Update personal best
                    if fitness < personal_best_fitness[i]:
                        personal_best[i] = particles[i].copy()
                        personal_best_fitness[i] = fitness

                        # Update global best
                        if fitness < global_best_fitness:
                            global_best = particles[i].copy()
                            global_best_fitness = fitness
                            print(f"  🎯 NEW GLOBAL BEST (Iteration {iteration + 1}): Error = {fitness:.2f}cm")

            # Update velocities and positions
            for i in range(n_particles):
                for param in particles[i]:
                    # Update velocity
                    r1, r2 = random.random(), random.random()

                    cognitive_velocity = c1 * r1 * (personal_best[i][param] - particles[i][param])
                    social_velocity = c2 * r2 * (global_best[param] - particles[i][param])

                    velocities[i][param] = w * velocities[i][param] + cognitive_velocity + social_velocity

                    # Update position
                    particles[i][param] += velocities[i][param]

                    # Apply bounds
                    if param == 'min_peak_distance':
                        particles[i][param] = max(bounds[param][0], min(bounds[param][1], int(particles[i][param])))
                    else:
                        particles[i][param] = max(bounds[param][0], min(bounds[param][1], particles[i][param]))

        print(f"\n=== PSO OPTIMIZATION COMPLETE ===")
        print(f"Best Parameters: {global_best}")
        print(f"Best Error: {global_best_fitness:.2f}cm")

        return {
            'best_parameters': global_best,
            'best_error': global_best_fitness,
            'optimization_method': 'PSO'
        }
    def optimize_parameters_bayesian(self, segments_dir: str, n_trials: int = 30) -> Dict:
        """
        Optimize parameters using Bayesian optimization (much faster than grid search).

        Args:
            segments_dir: Directory containing segment CSV files
            n_trials: Number of optimization trials (default: 30)

        Returns:
            Dictionary with best parameters and results
        """

        def objective_function(params):
            step_threshold, min_peak_distance, base_step_length_cm = params

            # Update parameters
            self.step_threshold = step_threshold
            self.min_peak_distance = int(min_peak_distance)

            # Process segments
            results = self.process_all_segments_optimization(segments_dir, base_step_length_cm)

            if not results:
                return float('inf')

            # Calculate total error (objective to minimize)
            total_error = sum(r['error_cm'] for r in results)
            return total_error

        # Parameter bounds
        bounds = [
            (0.2, 1.0),  # step_threshold
            (15, 80),  # min_peak_distance
            (40, 80)  # base_step_length_cm
        ]

        # Initial guess
        x0 = [0.5, 40, 60]

        print(f"Starting Bayesian optimization with {n_trials} trials...")

        # Run optimization
        result = minimize(
            objective_function,
            x0,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': n_trials}
        )

        best_params = {
            'step_threshold': result.x[0],
            'min_peak_distance': int(result.x[1]),
            'base_step_length_cm': result.x[2]
        }

        print(f"\n=== OPTIMIZATION COMPLETE ===")
        print(f"Best Parameters: {best_params}")
        print(f"Best Error: {result.fun:.2f}cm")

        return {
            'best_parameters': best_params,
            'best_error': result.fun,
            'optimization_success': result.success
        }
    def optimize_parameters_grid_search(self, segments_dir: str, param_ranges: Dict = None) -> Dict:
        """
        Optimize parameters using grid search to minimize distance estimation error.

        Args:
            segments_dir: Directory containing segment CSV files
            param_ranges: Dictionary of parameter ranges to search

        Returns:
            Dictionary with best parameters and results
        """
        if param_ranges is None:
            param_ranges = {
                'step_threshold': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                'min_peak_distance': [20, 30, 40, 50, 60],
                'base_step_length_cm': [50, 55, 60, 65, 70]
            }

        best_params = None
        best_error = float('inf')
        best_results = None

        total_combinations = (len(param_ranges['step_threshold']) *
                              len(param_ranges['min_peak_distance']) *
                              len(param_ranges['base_step_length_cm']))

        print(f"Starting grid search with {total_combinations} parameter combinations...")

        combination_count = 0

        for step_threshold in param_ranges['step_threshold']:
            for min_peak_distance in param_ranges['min_peak_distance']:
                for base_step_length_cm in param_ranges['base_step_length_cm']:
                    combination_count += 1
                    print(f"Testing combination {combination_count}/{total_combinations}: "
                          f"threshold={step_threshold}, distance={min_peak_distance}, "
                          f"step_length={base_step_length_cm}")

                    # Update parameters
                    self.step_threshold = step_threshold
                    self.min_peak_distance = min_peak_distance

                    # Process all segments with current parameters
                    results = self.process_all_segments_optimization(segments_dir, base_step_length_cm)

                    if results:
                        # Calculate total error
                        total_error = sum(r['error_cm'] for r in results)
                        avg_error_percentage = np.mean([r['error_percentage'] for r in results])

                        print(f"  Total Error: {total_error:.2f}cm, Avg Error: {avg_error_percentage:.2f}%")

                        # Update best parameters if error is lower
                        if total_error < best_error:
                            best_error = total_error
                            best_params = {
                                'step_threshold': step_threshold,
                                'min_peak_distance': min_peak_distance,
                                'base_step_length_cm': base_step_length_cm
                            }
                            best_results = results
                            print(f"  🎯 NEW BEST: Error = {total_error:.2f}cm")

        print(f"\n=== OPTIMIZATION COMPLETE ===")
        print(f"Best Parameters: {best_params}")
        print(f"Best Total Error: {best_error:.2f}cm")

        return {
            'best_parameters': best_params,
            'best_error': best_error,
            'best_results': best_results
        }

    def process_all_segments_optimization(self, segments_dir: str, base_step_length_cm: float) -> List[Dict]:
        """
        Process all segments for optimization (without plotting).

        Args:
            segments_dir: Directory containing segment CSV files
            base_step_length_cm: Base step length to use

        Returns:
            List of results for all segments
        """
        all_results = []
        segment_files = [f for f in os.listdir(segments_dir) if f.endswith('.csv')]
        segment_files.sort()

        for segment_file in segment_files:
            file_path = os.path.join(segments_dir, segment_file)
            results = self.process_segment_optimization(file_path, base_step_length_cm)

            if results:
                all_results.append(results)

        return all_results

    def process_segment_optimization(self, segment_file: str, base_step_length_cm: float) -> Dict:
        """
        Process a single segment for optimization (without plotting).

        Args:
            segment_file: Path to the segment CSV file
            base_step_length_cm: Base step length to use

        Returns:
            Dictionary containing step detection results
        """
        # Load segment data
        df = pd.read_csv(segment_file)

        # Extract accelerometer data
        accel_data = df[df['sensor_type'] == 'TYPE_ACCELEROMETER'].copy()

        if accel_data.empty:
            return {}

        # Calculate magnitude for each accelerometer reading
        magnitudes = []
        timestamps = []

        for _, row in accel_data.iterrows():
            mag = self.calculate_magnitude(row['x'], row['y'], row['z'])
            magnitudes.append(mag)
            timestamps.append(row['timestamp'])

        # Remove gravity component
        net_magnitudes = self.remove_gravity(magnitudes)

        # Apply filters
        filtered_magnitudes = self.apply_filters(net_magnitudes)

        # Detect peaks
        peaks = self.detect_peaks_enhanced(filtered_magnitudes, timestamps)

        # Extract step lengths
        step_lengths = [peak['step_length_samples'] for peak in peaks]

        # Categorize steps
        categorized_steps = self.categorize_step_length(step_lengths)

        # Estimate distance with custom base step length
        estimated_distance = self.estimate_distance_optimization(peaks, categorized_steps, base_step_length_cm)

        # Calculate real-world distance from grid coordinates
        segment_name = os.path.basename(segment_file)
        grid_coords = self.extract_grid_coordinates(segment_name)

        if grid_coords:
            start_x, start_y = grid_coords['start']
            end_x, end_y = grid_coords['end']

            real_start_x, real_start_y = self.convert_grid_to_real_world(start_x, start_y)
            real_end_x, real_end_y = self.convert_grid_to_real_world(end_x, end_y)

            real_distance = math.sqrt((real_end_x - real_start_x) ** 2 + (real_end_y - real_start_y) ** 2)
        else:
            real_distance = 0.0

        results = {
            'segment_file': segment_file,
            'num_peaks': len(peaks),
            'estimated_distance_cm': estimated_distance,
            'real_distance_cm': real_distance,
            'error_cm': abs(estimated_distance - real_distance),
            'error_percentage': (
                        abs(estimated_distance - real_distance) / real_distance * 100) if real_distance > 0 else 0
        }

        return results

    def estimate_distance_optimization(self, peaks: List[Dict], categorized_steps: List[Dict],
                                       base_step_length_cm: float) -> float:
        """
        Estimate total distance using dynamic step length with custom base step length.

        Args:
            peaks: List of detected peaks
            categorized_steps: List of categorized steps with weights
            base_step_length_cm: Custom base step length

        Returns:
            Estimated distance in cm
        """
        if len(peaks) != len(categorized_steps):
            return 0.0

        total_distance = 0.0

        for peak, step_info in zip(peaks, categorized_steps):
            # Apply weight based on step category
            weighted_step_length = base_step_length_cm * step_info['weight']
            total_distance += weighted_step_length

        return total_distance


    def optimize_parameters_random_search(self, segments_dir: str, n_trials: int = 50) -> Dict:
        """
        Random search optimization - often faster and more reliable than Bayesian.
        """
        import random

        best_params = None
        best_error = float('inf')

        print(f"Starting random search with {n_trials} trials...")

        for trial in range(n_trials):
            # Generate random parameters
            step_threshold = random.uniform(0.2, 1.0)
            min_peak_distance = random.randint(15, 80)
            base_step_length_cm = random.uniform(40, 80)

            if trial % 10 == 0:  # Print every 10th trial
                print(f"Trial {trial + 1}/{n_trials}")

            # Update parameters
            self.step_threshold = step_threshold
            self.min_peak_distance = min_peak_distance

            # Process segments
            results = self.process_all_segments_optimization(segments_dir, base_step_length_cm)

            if results:
                total_error = sum(r['error_cm'] for r in results)

                if total_error < best_error:
                    best_error = total_error
                    best_params = {
                        'step_threshold': step_threshold,
                        'min_peak_distance': min_peak_distance,
                        'base_step_length_cm': base_step_length_cm
                    }
                    print(f"  �� NEW BEST (Trial {trial + 1}): Error = {total_error:.2f}cm")

        return {
            'best_parameters': best_params,
            'best_error': best_error
        }

    import multiprocessing as mp
    from functools import partial

    def evaluate_single_trial(args):
        """
        Evaluate a single parameter combination for parallel processing.
        This function must be at module level (not inside a class).
        """
        trial_id, segments_dir, step_threshold, min_peak_distance, base_step_length_cm = args

        # Create a new instance for this trial
        step_detector = StepDetectionAlgorithm(
            step_threshold=step_threshold,
            min_peak_distance=min_peak_distance
        )

        # Process segments
        results = step_detector.process_all_segments_optimization(segments_dir, base_step_length_cm)

        if not results:
            return float('inf'), None

        total_error = sum(r['error_cm'] for r in results)
        return total_error, {
            'step_threshold': step_threshold,
            'min_peak_distance': min_peak_distance,
            'base_step_length_cm': base_step_length_cm
        }

    # Then modify your class method:
    def optimize_parameters_parallel(self, segments_dir: str, n_trials: int = 100) -> Dict:
        """
        Parallel optimization using multiple CPU cores.
        """
        import random
        import multiprocessing as mp

        # Prepare arguments for parallel processing
        args_list = []
        for trial_id in range(n_trials):
            step_threshold = random.uniform(0.2, 1.0)
            min_peak_distance = random.randint(15, 80)
            base_step_length_cm = random.uniform(40, 80)

            args = (trial_id, segments_dir, step_threshold, min_peak_distance, base_step_length_cm)
            args_list.append(args)

        # Use all available CPU cores
        n_cores = mp.cpu_count()
        print(f"Using {n_cores} CPU cores for parallel optimization...")

        with mp.Pool(n_cores) as pool:
            results = pool.map(evaluate_single_trial, args_list)

        # Find best result
        best_error = float('inf')
        best_params = None

        for error, params in results:
            if error < best_error:
                best_error = error
                best_params = params

        return {
            'best_parameters': best_params,
            'best_error': best_error
        }


    def save_optimized_parameters(self, best_params: Dict, output_file: str = "optimized_parameters.json"):
        """
        Save optimized parameters to a JSON file.

        Args:
            best_params: Dictionary containing the best optimized parameters
            output_file: Path to save the parameters
        """
        # Add metadata to the parameters
        params_with_metadata = {
            'optimized_parameters': best_params,
            'metadata': {
                'optimization_date': pd.Timestamp.now().isoformat(),
                'algorithm_version': '1.0',
                'description': 'Optimized parameters for step detection algorithm',
                'parameter_descriptions': {
                    'step_threshold': 'Threshold for step detection (0.2-1.0)',
                    'min_peak_distance': 'Minimum distance between peaks in samples (15-80)',
                    'base_step_length_cm': 'Base step length in centimeters (40-80)'
                }
            }
        }

        # Save to JSON file
        with open(output_file, 'w') as f:
            json.dump(params_with_metadata, f, indent=2)

        print(f"✅ Optimized parameters saved to: {output_file}")

        # Also save a simple version for easy loading
        simple_file = output_file.replace('.json', '_simple.json')
        with open(simple_file, 'w') as f:
            json.dump(best_params, f, indent=2)

        print(f"✅ Simple parameters saved to: {simple_file}")


    def load_optimized_parameters(self, input_file: str = "optimized_parameters.json") -> Dict:
        """
        Load optimized parameters from a JSON file.

        Args:
            input_file: Path to the parameters file

        Returns:
            Dictionary containing the optimized parameters
        """
        try:
            with open(input_file, 'r') as f:
                data = json.load(f)

            if 'optimized_parameters' in data:
                # Full format with metadata
                params = data['optimized_parameters']
                print(f"✅ Loaded optimized parameters from: {input_file}")
                print(f"   Optimization date: {data['metadata']['optimization_date']}")
            else:
                # Simple format
                params = data
                print(f"✅ Loaded simple parameters from: {input_file}")

            return params

        except FileNotFoundError:
            print(f"❌ Parameters file not found: {input_file}")
            return None
        except json.JSONDecodeError:
            print(f"❌ Invalid JSON format in: {input_file}")
            return None

def evaluate_single_trial(args):
    """
    Evaluate a single parameter combination for parallel processing.
    This function must be at module level (not inside a class).
    """
    trial_id, segments_dir, step_threshold, min_peak_distance, base_step_length_cm = args

    # Create a new instance for this trial
    step_detector = StepDetectionAlgorithm(
        step_threshold=step_threshold,
        min_peak_distance=min_peak_distance
    )

    # Process segments
    results = step_detector.process_all_segments_optimization(segments_dir, base_step_length_cm)

    if not results:
        return float('inf'), None

    total_error = sum(r['error_cm'] for r in results)
    return total_error, {
        'step_threshold': step_threshold,
        'min_peak_distance': min_peak_distance,
        'base_step_length_cm': base_step_length_cm
    }


def main():
    """
    Main function to demonstrate the step detection algorithm.
    """
    # Initialize the algorithm with initial parameters
    step_detector = StepDetectionAlgorithm(
        step_threshold=0.5,
        min_peak_distance=30,
        window_size=10
    )

    # Process all segments
    segments_dir = "combined_path_segments"
    output_dir = "step_detection_results"

    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(segments_dir):
        # STEP 1: Test with initial parameters first
        print("=" * 60)
        print("TESTING WITH INITIAL PARAMETERS")
        print("=" * 60)
        print(f"Initial Parameters:")
        print(f"  step_threshold: {step_detector.step_threshold}")
        print(f"  min_peak_distance: {step_detector.min_peak_distance}")
        print(f"  base_step_length_cm: 60.0 (default)")

        # Process with initial parameters
        initial_results = step_detector.process_all_segments(segments_dir, output_dir + "_initial")

        if initial_results:
            # Calculate initial error metrics
            total_estimated_distance = sum(r['estimated_distance_cm'] for r in initial_results)
            total_real_distance = sum(r['real_distance_cm'] for r in initial_results)
            total_error = sum(r['error_cm'] for r in initial_results)
            avg_error_percentage = np.mean([r['error_percentage'] for r in initial_results])

            print(f"\nINITIAL PARAMETERS RESULTS:")
            print(f"  Total Estimated Distance: {total_estimated_distance:.2f} cm")
            print(f"  Total Real Distance: {total_real_distance:.2f} cm")
            print(f"  Total Error: {total_error:.2f} cm")
            print(f"  Average Error Percentage: {avg_error_percentage:.2f}%")

            # Save initial results
            initial_summary_data = []
            for result in initial_results:
                initial_summary_data.append({
                    'segment': os.path.basename(result['segment_file']),
                    'num_steps': result['num_peaks'],
                    'estimated_distance_cm': result['estimated_distance_cm'],
                    'real_distance_cm': result['real_distance_cm'],
                    'error_cm': result['error_cm'],
                    'error_percentage': result['error_percentage']
                })

            initial_summary_df = pd.DataFrame(initial_summary_data)
            initial_summary_df.to_csv(os.path.join(output_dir + "_initial", 'initial_parameters_summary.csv'),
                                      index=False)
            print(
                f"  Initial results saved to {os.path.join(output_dir + '_initial', 'initial_parameters_summary.csv')}")

        # STEP 2: Run optimization
        print("\n" + "=" * 60)
        print("RUNNING PARAMETER OPTIMIZATION")
        print("=" * 60)

        # OPTION 1: Grid Search Optimization
        print("Running grid search optimization...")
        optimization_results = step_detector.optimize_parameters_parallel(segments_dir)

        # Apply best parameters
        best_params = optimization_results['best_parameters']
        step_detector.step_threshold = best_params['step_threshold']
        step_detector.min_peak_distance = best_params['min_peak_distance']

        print(f"\nApplied optimized parameters: {best_params}")

        # STEP 2.5: SAVE OPTIMIZED PARAMETERS
        print("\n" + "=" * 60)
        print("SAVING OPTIMIZED PARAMETERS")
        print("=" * 60)

        # Save the optimized parameters
        step_detector.save_optimized_parameters(best_params, "optimized_parameters.json")

        # Also save to the output directory
        output_params_file = os.path.join(output_dir, "optimized_parameters.json")
        step_detector.save_optimized_parameters(best_params, output_params_file)

        # STEP 3: Test with optimized parameters
        print("\n" + "=" * 60)
        print("TESTING WITH OPTIMIZED PARAMETERS")
        print("=" * 60)

        # Process with optimized parameters
        optimized_results = step_detector.process_all_segments(segments_dir, output_dir)

        # Save optimized results to CSV
        if optimized_results:
            optimized_summary_data = []
            for result in optimized_results:
                optimized_summary_data.append({
                    'segment': os.path.basename(result['segment_file']),
                    'num_steps': result['num_peaks'],
                    'estimated_distance_cm': result['estimated_distance_cm'],
                    'real_distance_cm': result['real_distance_cm'],
                    'error_cm': result['error_cm'],
                    'error_percentage': result['error_percentage']
                })

            optimized_summary_df = pd.DataFrame(optimized_summary_data)
            optimized_summary_df.to_csv(os.path.join(output_dir, 'optimized_parameters_summary.csv'), index=False)
            print(f"\nOptimized results saved to {os.path.join(output_dir, 'optimized_parameters_summary.csv')}")

        # STEP 4: Compare results
        if initial_results and optimized_results:
            print("\n" + "=" * 60)
            print("COMPARISON: INITIAL vs OPTIMIZED PARAMETERS")
            print("=" * 60)

            # Print parameter comparison
            print(f"INITIAL PARAMETERS:")
            print(f"  step_threshold: {0.5}")
            print(f"  min_peak_distance: {30}")
            print(f"  base_step_length_cm: {60.0}")

            print(f"\nOPTIMIZED PARAMETERS:")
            print(f"  step_threshold: {best_params['step_threshold']}")
            print(f"  min_peak_distance: {best_params['min_peak_distance']}")
            print(f"  base_step_length_cm: {best_params['base_step_length_cm']}")

            print(f"\nPERFORMANCE COMPARISON:")
            # Calculate comparison metrics
            initial_total_error = sum(r['error_cm'] for r in initial_results)
            optimized_total_error = sum(r['error_cm'] for r in optimized_results)
            initial_avg_error = np.mean([r['error_percentage'] for r in initial_results])
            optimized_avg_error = np.mean([r['error_percentage'] for r in optimized_results])

            error_improvement = initial_total_error - optimized_total_error
            percentage_improvement = ((
                                              initial_total_error - optimized_total_error) / initial_total_error * 100) if initial_total_error > 0 else 0

            print(f"  Initial Parameters Error: {initial_total_error:.2f} cm")
            print(f"  Optimized Parameters Error: {optimized_total_error:.2f} cm")
            print(f"  Error Improvement: {error_improvement:.2f} cm ({percentage_improvement:.1f}%)")
            print(f"  Initial Average Error: {initial_avg_error:.2f}%")
            print(f"  Optimized Average Error: {optimized_avg_error:.2f}%")

            if error_improvement > 0:
                print(f"\n✅ OPTIMIZATION SUCCESSFUL: Error reduced by {error_improvement:.2f} cm")
            else:
                print(f"\n⚠️  OPTIMIZATION DID NOT IMPROVE: Error increased by {abs(error_improvement):.2f} cm")

    else:
        print(f"Segments directory '{segments_dir}' not found!")


if __name__ == "__main__":
    main()