def test_imports():
    try:
        import numpy as np
        import pytest
        from stream_analysis.density_profile_gap_detections import (
            log_difference_detection,
            apply_box_car_multiple_times,
            clean_up_stream_profiles,
            get_profile,
            counts_only_filter,
            apply_filter,
            counts_threshold_given_signal_to_noise_ratio,
            noise_log_counts,
            median_box_car,
            zscores,
            cross_correlation_shift,
        )
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")