import math

import pandas as pd
import re
import os


def parse_heading_markers(file_path):
    """
    Parse heading markers and identify completed path segments.

    Args:
        file_path (str): Path to the IMU data file

    Returns:
        list: List of completed path segments with their markers
    """
    heading_markers = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            # Check if this is a heading marker
            if 'turned to a New Heading TP' in line:
                # Extract heading information
                heading_match = re.search(
                    r'current point:\s*([\d.]+)\s*([\d.]+)\s*heading point:\s*([\d.]+)\s*([\d.]+)', line)
                if heading_match:
                    current_x, current_y, heading_x, heading_y = map(float, heading_match.groups())

                    # Extract timestamp (first part before tab)
                    parts = line.split('\t')
                    timestamp = int(parts[0])

                    marker = {
                        'timestamp': timestamp,
                        'current_point': (current_x, current_y),
                        'heading_point': (heading_x, heading_y),
                        'line': line
                    }
                    heading_markers.append(marker)

    return heading_markers


def create_all_segments(heading_markers):
    """
    Create segments for ALL consecutive heading markers (before validation).

    Args:
        heading_markers (list): List of heading markers

    Returns:
        list: List of all segments
    """
    all_segments = []

    print("\n=== CREATING ALL SEGMENTS ===")
    for i in range(len(heading_markers) - 1):
        current_marker = heading_markers[i]
        next_marker = heading_markers[i + 1]

        segment = {
            'segment_id': i,
            'start_marker': current_marker,
            'end_marker': next_marker,
            'start_timestamp': current_marker['timestamp'],
            'end_timestamp': next_marker['timestamp'],
            'path': f"({current_marker['current_point'][0]:.0f},{current_marker['current_point'][1]:.0f}) -> ({next_marker['heading_point'][0]:.0f},{next_marker['heading_point'][1]:.0f})",
            'previous_point': current_marker['current_point'],
            'heading_point': current_marker['heading_point'],
            'current_point': next_marker['current_point']
        }
        all_segments.append(segment)
        print(
            f"Segment {i}: Previous {current_marker['current_point']} -> Heading {current_marker['heading_point']} -> Current {next_marker['current_point']}")

    return all_segments


def validate_segments(all_segments):
    """
    Validate segments based on heading point = current point rule.

    Args:
        all_segments (list): List of all segments

    Returns:
        list: List of valid segments
    """
    valid_segments = []

    print("\n=== VALIDATION PROCESS ===")
    for segment in all_segments:
        heading_point = segment['heading_point']
        current_point = segment['current_point']

        is_valid = heading_point == current_point

        if is_valid:
            valid_segments.append(segment)
            print(f"✅ VALID: Segment {segment['segment_id']} - Heading {heading_point} = Current {current_point}")
        else:
            print(f"❌ INVALID: Segment {segment['segment_id']} - Heading {heading_point} ≠ Current {current_point}")

    print(f"\n=== SUMMARY ===")
    print(f"Total segments: {len(all_segments)}")
    print(f"Valid segments: {len(valid_segments)}")
    print(f"Invalid segments: {len(all_segments) - len(valid_segments)}")

    return valid_segments


def extract_imu_data_for_segments(file_path, valid_segments):
    """
    Extract IMU data for each valid path segment.

    Args:
        file_path (str): Path to the IMU data file
        valid_segments (list): List of valid path segments

    Returns:
        list: List of segments with their IMU data
    """
    segments_with_data = []

    for segment in valid_segments:
        start_time = segment['start_timestamp']
        end_time = segment['end_timestamp']

        segment_data = {
            'segment_id': segment['segment_id'],
            'start_marker': segment['start_marker'],
            'end_marker': segment['end_marker'],
            'path': segment['path'],
            'start_timestamp': start_time,
            'end_timestamp': end_time,
            'imu_data': []
        }

        # Read file and extract IMU data between timestamps
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')
                if len(parts) < 2:
                    continue

                try:
                    timestamp = int(parts[0])

                    # Check if timestamp is within segment range (inclusive)
                    if start_time <= timestamp <= end_time:
                        # Skip heading marker lines
                        if 'turned to a New Heading TP' not in line:
                            sensor_type = parts[1]
                            sensor_data = parts[2:]

                            # Parse IMU data
                            imu_entry = parse_imu_line(timestamp, sensor_type, sensor_data)
                            if imu_entry:
                                segment_data['imu_data'].append(imu_entry)
                except ValueError:
                    continue

        segments_with_data.append(segment_data)

    return segments_with_data


def parse_imu_line(timestamp, sensor_type, sensor_data):
    """
    Parse a single IMU data line.

    Args:
        timestamp (int): Timestamp
        sensor_type (str): Type of sensor
        sensor_data (list): Sensor data values

    Returns:
        dict: Parsed IMU entry or None if invalid
    """
    try:
        if sensor_type in ['TYPE_ACCELEROMETER', 'TYPE_MAGNETIC_FIELD', 'TYPE_LINEAR_ACCELERATION']:
            if len(sensor_data) >= 3:
                values = [float(x) for x in sensor_data[:3]]
                return {
                    'timestamp': timestamp,
                    'sensor_type': sensor_type,
                    'x': values[0],
                    'y': values[1],
                    'z': values[2]
                }
        elif sensor_type == 'TYPE_ROTATION_VECTOR':
            if len(sensor_data) >= 4:
                values = [float(x) for x in sensor_data[:4]]
                return {
                    'timestamp': timestamp,
                    'sensor_type': sensor_type,
                    'x': values[0],
                    'y': values[1],
                    'z': values[2],
                    'w': values[3]
                }
    except ValueError:
        return None

    return None


def save_segments_to_csv(segments_with_data, output_dir='completed_path_segments'):
    """
    Save each valid path segment to a separate CSV file.

    Args:
        segments_with_data (list): List of segments with IMU data
        output_dir (str): Directory to save CSV files
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== SAVING VALID SEGMENTS ===")
    for segment in segments_with_data:
        if not segment['imu_data']:
            continue

        # Create filename based on segment info
        segment_id = segment['segment_id']
        start_point = segment['start_marker']['current_point']
        end_point = segment['end_marker']['current_point']

        filename = f"segment_{segment_id:03d}_{start_point[0]:.0f}_{start_point[1]:.0f}_to_{end_point[0]:.0f}_{end_point[1]:.0f}.csv"
        filepath = os.path.join(output_dir, filename)

        # Convert to DataFrame and save
        df = pd.DataFrame(segment['imu_data'])
        df.to_csv(filepath, index=False)
        print(
            f"✅ Saved VALID segment {segment_id}: {start_point} -> {end_point} ({len(segment['imu_data'])} IMU records)")

    print(f"\nTotal valid segments saved: {len([s for s in segments_with_data if s['imu_data']])}")


def save_combined_segments_to_csv(combined_segments, output_dir='combined_path_segments'):
    """
    Save combined segments to CSV files.

    Args:
        combined_segments: List of combined segments
        output_dir: Directory to save CSV files
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== SAVING COMBINED SEGMENTS ===")
    for segment in combined_segments:
        if not segment['imu_data']:
            continue

        # Create filename based on segment info
        segment_id = segment['segment_id']
        start_point = segment['start_marker']['current_point']
        end_point = segment['end_marker']['current_point']

        # Handle combined segment IDs - FIXED: Remove double "combined_" prefix
        if isinstance(segment_id, str) and segment_id.startswith('combined_'):
            # Extract the numeric part from combined_0_2 format
            numeric_part = segment_id.replace('combined_', '')
            filename = f"combined_{numeric_part}_{start_point[0]:.0f}_{start_point[1]:.0f}_to_{end_point[0]:.0f}_{end_point[1]:.0f}.csv"
        else:
            filename = f"segment_{segment_id:03d}_{start_point[0]:.0f}_{start_point[1]:.0f}_to_{end_point[0]:.0f}_{end_point[1]:.0f}.csv"

        filepath = os.path.join(output_dir, filename)

        # Convert to DataFrame and save
        df = pd.DataFrame(segment['imu_data'])
        df.to_csv(filepath, index=False)

        if 'num_combined' in segment:
            print(f"✅ Saved COMBINED segment {segment_id}: {start_point} -> {end_point} "
                  f"({len(segment['imu_data'])} IMU records, {segment['num_combined']} original segments)")
        else:
            print(f"✅ Saved SINGLE segment {segment_id}: {start_point} -> {end_point} "
                  f"({len(segment['imu_data'])} IMU records)")

    print(f"\nTotal combined segments saved: {len([s for s in combined_segments if s['imu_data']])}")
def calculate_direction(start_point, end_point):
    """
    Calculate the direction vector between two points.

    Args:
        start_point: Tuple of (x, y) coordinates
        end_point: Tuple of (x, y) coordinates

    Returns:
        Tuple of (dx, dy) representing the direction vector
    """
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]

    # Normalize to get unit direction vector
    magnitude = math.sqrt(dx ** 2 + dy ** 2)
    if magnitude > 0:
        dx = dx / magnitude
        dy = dy / magnitude

    return dx, dy


def are_same_direction(dir1, dir2, tolerance=0.1):
    """
    Check if two direction vectors are approximately the same.
    
    Args:
        dir1: First direction vector (dx, dy)
        dir2: Second direction vector (dx, dy)
        tolerance: Tolerance for considering directions the same
        
    Returns:
        True if directions are similar, False otherwise
    """
    # Calculate dot product
    dot_product = dir1[0] * dir2[0] + dir1[1] * dir2[1]
    
    # If dot product is close to 1, directions are similar
    # If dot product is close to -1, directions are opposite
    # We want directions to be similar (positive dot product)
    return dot_product > (1 - tolerance)


def combine_consecutive_segments(segments_with_data):
    """
    Combine consecutive segments that are moving in the same direction.

    Args:
        segments_with_data: List of segments with IMU data

    Returns:
        List of combined segments
    """
    if not segments_with_data:
        return []

    combined_segments = []
    current_group = [segments_with_data[0]]

    print("\n=== COMBINING CONSECUTIVE SEGMENTS ===")

    for i in range(1, len(segments_with_data)):
        current_segment = segments_with_data[i]
        last_segment = current_group[-1]

        # Calculate directions
        last_start = last_segment['start_marker']['current_point']
        last_end = last_segment['end_marker']['current_point']
        current_start = current_segment['start_marker']['current_point']
        current_end = current_segment['end_marker']['current_point']

        last_direction = calculate_direction(last_start, last_end)
        current_direction = calculate_direction(current_start, current_end)

        # Check if segments are consecutive and moving in same direction
        segments_consecutive = last_end == current_start
        same_direction = are_same_direction(last_direction, current_direction)

        print(f"Segment {last_segment['segment_id']} -> {current_segment['segment_id']}:")
        print(f"  Last: {last_start} -> {last_end} (dir: {last_direction})")
        print(f"  Current: {current_start} -> {current_end} (dir: {current_direction})")
        print(f"  Consecutive: {segments_consecutive}, Same direction: {same_direction}")

        if segments_consecutive and same_direction:
            # Add to current group
            current_group.append(current_segment)
            print(f"  ✅ COMBINING segments {[s['segment_id'] for s in current_group]}")
        else:
            # Save current group and start new one
            if len(current_group) > 1:
                combined_segment = combine_segment_group(current_group)
                combined_segments.append(combined_segment)
                print(f"  💾 SAVED combined segment: {combined_segment['segment_id']}")
            else:
                combined_segments.append(current_group[0])
                print(f"  💾 SAVED single segment: {current_group[0]['segment_id']}")

            current_group = [current_segment]

    # Handle the last group
    if len(current_group) > 1:
        combined_segment = combine_segment_group(current_group)
        combined_segments.append(combined_segment)
        print(f"  💾 SAVED final combined segment: {combined_segment['segment_id']}")
    else:
        combined_segments.append(current_group[0])
        print(f"  �� SAVED final single segment: {current_group[0]['segment_id']}")

    print(f"\n=== COMBINATION SUMMARY ===")
    print(f"Original segments: {len(segments_with_data)}")
    print(f"Combined segments: {len(combined_segments)}")

    return combined_segments


def combine_segment_group(segment_group):
    """
    Combine a group of consecutive segments into one segment.

    Args:
        segment_group: List of segments to combine

    Returns:
        Combined segment
    """
    if not segment_group:
        return None

    # Use the first segment as base
    combined = segment_group[0].copy()

    # Update segment ID to indicate it's combined
    segment_ids = [s['segment_id'] for s in segment_group]
    combined['segment_id'] = f"combined_{min(segment_ids)}_{max(segment_ids)}"

    # Update start and end markers
    combined['start_marker'] = segment_group[0]['start_marker']
    combined['end_marker'] = segment_group[-1]['end_marker']
    combined['start_timestamp'] = segment_group[0]['start_timestamp']
    combined['end_timestamp'] = segment_group[-1]['end_timestamp']

    # Combine IMU data from all segments
    combined_imu_data = []
    for segment in segment_group:
        combined_imu_data.extend(segment['imu_data'])

    combined['imu_data'] = combined_imu_data

    # Update path description
    start_point = combined['start_marker']['current_point']
    end_point = combined['end_marker']['current_point']
    combined['path'] = f"({start_point[0]:.0f},{start_point[1]:.0f}) -> ({end_point[0]:.0f},{end_point[1]:.0f})"

    # Add information about original segments
    combined['original_segments'] = segment_ids
    combined['num_combined'] = len(segment_group)

    return combined


def save_combined_segments_to_csv(combined_segments, output_dir='combined_path_segments'):
    """
    Save combined segments to CSV files.

    Args:
        combined_segments: List of combined segments
        output_dir: Directory to save CSV files
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== SAVING COMBINED SEGMENTS ===")
    for segment in combined_segments:
        if not segment['imu_data']:
            continue

        # Create filename based on segment info
        segment_id = segment['segment_id']
        start_point = segment['start_marker']['current_point']
        end_point = segment['end_marker']['current_point']

        # Handle combined segment IDs - FIXED: Remove double "combined_" prefix
        if isinstance(segment_id, str) and segment_id.startswith('combined_'):
            # Extract the numeric part from combined_0_2 format
            numeric_part = segment_id.replace('combined_', '')
            filename = f"combined_{numeric_part}_{start_point[0]:.0f}_{start_point[1]:.0f}_to_{end_point[0]:.0f}_{end_point[1]:.0f}.csv"
        else:
            filename = f"segment_{segment_id:03d}_{start_point[0]:.0f}_{start_point[1]:.0f}_to_{end_point[0]:.0f}_{end_point[1]:.0f}.csv"

        filepath = os.path.join(output_dir, filename)

        # Convert to DataFrame and save
        df = pd.DataFrame(segment['imu_data'])
        df.to_csv(filepath, index=False)

        if 'num_combined' in segment:
            print(f"✅ Saved COMBINED segment {segment_id}: {start_point} -> {end_point} "
                  f"({len(segment['imu_data'])} IMU records, {segment['num_combined']} original segments)")
        else:
            print(f"✅ Saved SINGLE segment {segment_id}: {start_point} -> {end_point} "
                  f"({len(segment['imu_data'])} IMU records)")

    print(f"\nTotal combined segments saved: {len([s for s in combined_segments if s['imu_data']])}")


def extract_grid_coordinates(segment_name: str):
    """
    Extract grid coordinates from segment filename.

    Args:
        segment_name: Segment filename

    Returns:
        Dictionary with start and end coordinates, or None if parsing fails
    """
    try:
        # Remove .csv extension
        name_without_ext = segment_name.replace('.csv', '')

        # Handle different filename formats
        if name_without_ext.startswith('combined_'):
            # Format: combined_0_2_319_143_to_319_317
            parts = name_without_ext.split('_')
            if len(parts) >= 7:
                start_x = float(parts[3])
                start_y = float(parts[4])
                end_x = float(parts[6])
                end_y = float(parts[7])

                return {
                    'start': (start_x, start_y),
                    'end': (end_x, end_y)
                }
        else:
            # Format: segment_000_319_143_to_319_158
            parts = name_without_ext.split('_')
            if len(parts) >= 6:
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
def main():
    """
    Main function to process IMU data and extract completed path segments.
    """
    # Replace with your actual file path
    file_path = '../assets/PDR_optimization_data_path1.txt'

    try:
        print("Parsing heading markers...")
        heading_markers = parse_heading_markers(file_path)
        print(f"Found {len(heading_markers)} heading markers")

        print("\nCreating all segments...")
        all_segments = create_all_segments(heading_markers)
        print(f"Created {len(all_segments)} segments")

        print("\nValidating segments...")
        valid_segments = validate_segments(all_segments)
        print(f"Found {len(valid_segments)} valid segments")

        print("\nExtracting IMU data for valid segments...")
        segments_with_data = extract_imu_data_for_segments(file_path, valid_segments)

        # NEW: Combine consecutive segments moving in same direction
        print("\nCombining consecutive segments moving in same direction...")
        combined_segments = combine_consecutive_segments(segments_with_data)

        # Save original segments to CSV files
        print("\nSaving original valid segments to CSV files...")
        save_segments_to_csv(segments_with_data)

        # Save combined segments to CSV files
        print("\nSaving combined segments to CSV files...")
        save_combined_segments_to_csv(combined_segments)

        print("\nProcessing complete!")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please update the file_path variable with the correct path.")
    except Exception as e:
        print(f"Error processing file: {e}")



if __name__ == "__main__":
    main()