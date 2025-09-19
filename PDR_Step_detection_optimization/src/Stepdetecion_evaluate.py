import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple, Optional
import math
from stepdetection import StepDetectionAlgorithm


class StepDetectionEvaluator:
    """
    Evaluator class that uses optimized parameters to test step detection
    performance on different datasets and compare with ground truth.
    """

    def __init__(self,
                 optimized_params: Dict,
                 map_width_cm: float = 2643.49,
                 map_height_cm: float = 3264.33,
                 grid_width: int = 928,
                 grid_height: int = 1167):
        """
        Initialize evaluator with optimized parameters.

        Args:
            optimized_params: Dictionary containing optimized parameters
            map_width_cm: Real-world map width in cm
            map_height_cm: Real-world map height in cm
            grid_width: Grid width in pixels
            grid_height: Grid height in pixels
        """
        self.optimized_params = optimized_params
        self.map_width_cm = map_width_cm
        self.map_height_cm = map_height_cm
        self.grid_width = grid_width
        self.grid_height = grid_height

        # Create step detector with optimized parameters
        self.step_detector = StepDetectionAlgorithm(
            step_threshold=optimized_params['step_threshold'],
            min_peak_distance=optimized_params['min_peak_distance'],
            window_size=10,
            map_width_cm=map_width_cm,
            map_height_cm=map_height_cm,
            grid_width=grid_width,
            grid_height=grid_height
        )

        # Store evaluation results
        self.evaluation_results = []

    def evaluate_dataset(self,
                         segments_dir: str,
                         dataset_name: str = "Test Dataset",
                         output_dir: str = "evaluation_results") -> Dict:
        """
        Evaluate step detection performance on a dataset.

        Args:
            segments_dir: Directory containing segment CSV files
            dataset_name: Name of the dataset for reporting
            output_dir: Directory to save evaluation results

        Returns:
            Dictionary containing evaluation results
        """
        print(f"\n{'=' * 60}")
        print(f"EVALUATING DATASET: {dataset_name}")
        print(f"{'=' * 60}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Process all segments
        segment_files = [f for f in os.listdir(segments_dir) if f.endswith('.csv')]
        segment_files.sort()

        dataset_results = []
        total_estimated_distance = 0
        total_real_distance = 0
        total_steps = 0

        print(f"Processing {len(segment_files)} segments...")

        for i, segment_file in enumerate(segment_files):
            file_path = os.path.join(segments_dir, segment_file)

            # Process segment with optimized parameters
            result = self._process_segment_with_optimized_params(
                file_path,
                self.optimized_params['base_step_length_cm']
            )

            if result:
                dataset_results.append(result)
                total_estimated_distance += result['estimated_distance_cm']
                total_real_distance += result['real_distance_cm']
                total_steps += result['num_steps']

                print(f"Segment {i + 1:2d}: {result['num_steps']:2d} steps, "
                      f"Est: {result['estimated_distance_cm']:6.1f}cm, "
                      f"Real: {result['real_distance_cm']:6.1f}cm, "
                      f"Error: {result['error_cm']:6.1f}cm ({result['error_percentage']:5.1f}%)")

        # Calculate overall statistics
        if dataset_results:
            avg_error_percentage = np.mean([r['error_percentage'] for r in dataset_results])
            std_error_percentage = np.std([r['error_percentage'] for r in dataset_results])
            total_error = abs(total_estimated_distance - total_real_distance)
            overall_error_percentage = (total_error / total_real_distance * 100) if total_real_distance > 0 else 0

            # Create summary
            summary = {
                'dataset_name': dataset_name,
                'num_segments': len(dataset_results),
                'total_steps': total_steps,
                'total_estimated_distance_cm': total_estimated_distance,
                'total_real_distance_cm': total_real_distance,
                'total_error_cm': total_error,
                'overall_error_percentage': overall_error_percentage,
                'avg_error_percentage': avg_error_percentage,
                'std_error_percentage': std_error_percentage,
                'segment_results': dataset_results,
                'optimized_params': self.optimized_params
            }

            # Save detailed results
            self._save_evaluation_results(summary, output_dir, dataset_name)

            # Print summary
            self._print_evaluation_summary(summary)

            # Store for comparison
            self.evaluation_results.append(summary)

            return summary
        else:
            print("No valid segments found for evaluation.")
            return {}

    def _process_segment_with_optimized_params(self,
                                               segment_file: str,
                                               base_step_length_cm: float) -> Optional[Dict]:
        """
        Process a single segment using optimized parameters.

        Args:
            segment_file: Path to the segment CSV file
            base_step_length_cm: Base step length from optimized parameters

        Returns:
            Dictionary containing step detection results
        """
        try:
            # Load segment data
            df = pd.read_csv(segment_file)

            # Extract accelerometer data
            accel_data = df[df['sensor_type'] == 'TYPE_ACCELEROMETER'].copy()

            if accel_data.empty:
                return None

            # Calculate magnitude for each accelerometer reading
            magnitudes = []
            timestamps = []

            for _, row in accel_data.iterrows():
                mag = self.step_detector.calculate_magnitude(row['x'], row['y'], row['z'])
                magnitudes.append(mag)
                timestamps.append(row['timestamp'])

            # Remove gravity component
            net_magnitudes = self.step_detector.remove_gravity(magnitudes)

            # Apply filters
            filtered_magnitudes = self.step_detector.apply_filters(net_magnitudes)

            # Detect peaks
            peaks = self.step_detector.detect_peaks_enhanced(filtered_magnitudes, timestamps)

            # Extract step lengths
            step_lengths = [peak['step_length_samples'] for peak in peaks]

            # Categorize steps
            categorized_steps = self.step_detector.categorize_step_length(step_lengths)

            # Estimate distance with optimized base step length
            estimated_distance = self.step_detector.estimate_distance_optimization(
                peaks, categorized_steps, base_step_length_cm
            )

            # Calculate real-world distance from grid coordinates
            segment_name = os.path.basename(segment_file)
            grid_coords = self.step_detector.extract_grid_coordinates(segment_name)

            if grid_coords:
                start_x, start_y = grid_coords['start']
                end_x, end_y = grid_coords['end']

                # Convert to real-world coordinates
                start_real_x, start_real_y = self.step_detector.convert_grid_to_real_world(start_x, start_y)
                end_real_x, end_real_y = self.step_detector.convert_grid_to_real_world(end_x, end_y)

                # Calculate real distance
                real_distance = math.sqrt((end_real_x - start_real_x) ** 2 + (end_real_y - start_real_y) ** 2)

                # Calculate error
                error_cm = abs(estimated_distance - real_distance)
                error_percentage = (error_cm / real_distance * 100) if real_distance > 0 else 0

                return {
                    'segment_name': segment_name,
                    'num_steps': len(peaks),
                    'estimated_distance_cm': estimated_distance,
                    'real_distance_cm': real_distance,
                    'error_cm': error_cm,
                    'error_percentage': error_percentage,
                    'start_coords': (start_x, start_y),
                    'end_coords': (end_x, end_y),
                    'start_real_coords': (start_real_x, start_real_y),
                    'end_real_coords': (end_real_x, end_real_y),
                    'peaks': peaks,
                    'step_lengths': step_lengths
                }

        except Exception as e:
            print(f"Error processing {segment_file}: {e}")
            return None

    def _save_evaluation_results(self, summary: Dict, output_dir: str, dataset_name: str):
        """
        Save evaluation results to files.

        Args:
            summary: Evaluation summary dictionary
            output_dir: Output directory
            dataset_name: Name of the dataset
        """
        # Save summary to JSON
        import json
        summary_file = os.path.join(output_dir, f"{dataset_name}_summary.json")

        # Convert numpy types to native Python types for JSON serialization
        json_summary = {}
        for key, value in summary.items():
            if key != 'segment_results':  # Skip detailed results for JSON
                if isinstance(value, np.integer):
                    json_summary[key] = int(value)
                elif isinstance(value, np.floating):
                    json_summary[key] = float(value)
                else:
                    json_summary[key] = value

        with open(summary_file, 'w') as f:
            json.dump(json_summary, f, indent=2)

        # Save detailed segment results to CSV
        if summary['segment_results']:
            results_df = pd.DataFrame(summary['segment_results'])
            csv_file = os.path.join(output_dir, f"{dataset_name}_detailed_results.csv")
            results_df.to_csv(csv_file, index=False)

        print(f"✅ Results saved to {output_dir}/")

    def _print_evaluation_summary(self, summary: Dict):
        """
        Print evaluation summary.

        Args:
            summary: Evaluation summary dictionary
        """
        print(f"\n{'=' * 60}")
        print(f"EVALUATION SUMMARY: {summary['dataset_name']}")
        print(f"{'=' * 60}")
        print(f"Number of segments: {summary['num_segments']}")
        print(f"Total steps detected: {summary['total_steps']}")
        print(f"Total estimated distance: {summary['total_estimated_distance_cm']:.1f} cm")
        print(f"Total real distance: {summary['total_real_distance_cm']:.1f} cm")
        print(f"Total error: {summary['total_error_cm']:.1f} cm")
        print(f"Overall error percentage: {summary['overall_error_percentage']:.2f}%")
        print(
            f"Average error per segment: {summary['avg_error_percentage']:.2f}% ± {summary['std_error_percentage']:.2f}%")
        print(f"\nOptimized Parameters Used:")
        for param, value in summary['optimized_params'].items():
            print(f"  {param}: {value}")

    def compare_datasets(self, dataset_results: List[Dict]) -> Dict:
        """
        Compare performance across multiple datasets.

        Args:
            dataset_results: List of evaluation results from different datasets

        Returns:
            Dictionary containing comparison results
        """
        print(f"\n{'=' * 60}")
        print("DATASET COMPARISON")
        print(f"{'=' * 60}")

        comparison_data = []

        for result in dataset_results:
            comparison_data.append({
                'Dataset': result['dataset_name'],
                'Segments': result['num_segments'],
                'Total Steps': result['total_steps'],
                'Est. Distance (cm)': f"{result['total_estimated_distance_cm']:.1f}",
                'Real Distance (cm)': f"{result['total_real_distance_cm']:.1f}",
                'Total Error (cm)': f"{result['total_error_cm']:.1f}",
                'Overall Error (%)': f"{result['overall_error_percentage']:.2f}",
                'Avg Error (%)': f"{result['avg_error_percentage']:.2f}",
                'Std Error (%)': f"{result['std_error_percentage']:.2f}"
            })

        # Create comparison table
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))

        # Save comparison
        comparison_df.to_csv("evaluation_results/dataset_comparison.csv", index=False)

        return {
            'comparison_table': comparison_df,
            'datasets': dataset_results
        }

    def plot_performance_comparison(self, output_dir: str = "evaluation_results"):
        """
        Create performance comparison plots.

        Args:
            output_dir: Directory to save plots
        """
        if not self.evaluation_results:
            print("No evaluation results to plot.")
            return

        os.makedirs(output_dir, exist_ok=True)

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Step Detection Performance Comparison Across Datasets', fontsize=16)

        # Extract data for plotting
        dataset_names = [r['dataset_name'] for r in self.evaluation_results]
        error_percentages = [r['overall_error_percentage'] for r in self.evaluation_results]
        avg_errors = [r['avg_error_percentage'] for r in self.evaluation_results]
        total_steps = [r['total_steps'] for r in self.evaluation_results]
        total_distances = [r['total_real_distance_cm'] for r in self.evaluation_results]

        # Plot 1: Overall Error Percentage
        axes[0, 0].bar(dataset_names, error_percentages, color='skyblue')
        axes[0, 0].set_title('Overall Error Percentage')
        axes[0, 0].set_ylabel('Error (%)')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Plot 2: Average Error per Segment
        axes[0, 1].bar(dataset_names, avg_errors, color='lightcoral')
        axes[0, 1].set_title('Average Error per Segment')
        axes[0, 1].set_ylabel('Error (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Plot 3: Total Steps Detected
        axes[1, 0].bar(dataset_names, total_steps, color='lightgreen')
        axes[1, 0].set_title('Total Steps Detected')
        axes[1, 0].set_ylabel('Number of Steps')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Plot 4: Total Distance
        axes[1, 1].bar(dataset_names, total_distances, color='gold')
        axes[1, 1].set_title('Total Real Distance')
        axes[1, 1].set_ylabel('Distance (cm)')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✅ Performance plots saved to {output_dir}/performance_comparison.png")


def main():
    """
    Example usage of the StepDetectionEvaluator.
    """
    # Example optimized parameters (replace with your actual optimized parameters)
    optimized_params = {
        'step_threshold': 0.45,
        'min_peak_distance': 35,
        'base_step_length_cm': 62.5
    }

    # Create evaluator
    evaluator = StepDetectionEvaluator(optimized_params)

    # Evaluate on different datasets
    datasets = [
        {
            'name': 'Path1_Test',
            'dir': 'completed_path_segments_path1',
            'description': 'Test segments from Path 1'
        }
    ]

    # Evaluate each dataset
    for dataset in datasets:
        if os.path.exists(dataset['dir']):
            evaluator.evaluate_dataset(
                segments_dir=dataset['dir'],
                dataset_name=dataset['name']
            )
        else:
            print(f"Dataset directory {dataset['dir']} not found.")

    # Compare datasets
    if len(evaluator.evaluation_results) > 1:
        evaluator.compare_datasets(evaluator.evaluation_results)
        evaluator.plot_performance_comparison()

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()