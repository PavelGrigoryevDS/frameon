import pytest
import pandas as pd
import numpy as np
from frameon.api.utils import (
    analyze_join_keys,
    find_inconsistent_mappings,
    haversine_vectorized
)

@pytest.fixture
def sample_dataframes():
    """Fixture providing sample DataFrames for testing"""
    df1 = pd.DataFrame({
        'id': [1, 2, 3],
        'value': ['a', 'b', 'c']
    })
    df2 = pd.DataFrame({
        'id': [2, 3, 4],
        'value': ['x', 'y', 'z']
    })
    return df1, df2

class TestAnalyzeJoinKeys:
    """Tests for analyze_join_keys function"""
    
    def test_basic_join_analysis(self):
        """Test basic join analysis between two simple DataFrames"""
        df1 = pd.DataFrame({'id': [1, 2, 3], 'value': ['a', 'b', 'c']})
        df2 = pd.DataFrame({'id': [2, 3, 4], 'value': ['x', 'y', 'z']})
        
        # We'll need to capture stdout to verify the output
        # For now just verify it runs without errors
        analyze_join_keys(df1, df2, on='id')

    def test_empty_dataframe(self):
        """Test with empty DataFrame inputs"""
        df1 = pd.DataFrame({'id': [], 'value': []})
        df2 = pd.DataFrame({'id': [1, 2], 'value': ['a', 'b']})
        
        with pytest.raises(ValueError, match="Input DataFrames cannot be empty"):
            analyze_join_keys(df1, df2, on='id')
        
        with pytest.raises(ValueError, match="Input DataFrames cannot be empty"):
            analyze_join_keys(df2, df1, on='id')

    def test_invalid_key_configuration(self):
        """Test invalid key configurations"""
        df1 = pd.DataFrame({'id': [1, 2], 'value': ['a', 'b']})
        df2 = pd.DataFrame({'id': [2, 3], 'value': ['x', 'y']})
        
        # Missing right_on
        with pytest.raises(ValueError, match="Must specify either 'on'"):
            analyze_join_keys(df1, df2, left_on='id')
        
        # Mismatched key counts
        with pytest.raises(ValueError, match="Mismatched key counts"):
            analyze_join_keys(df1, df2, left_on=['id', 'value'], right_on=['id'])

    def test_all_nan_keys(self):
        """Test with all NaN keys"""
        df1 = pd.DataFrame({'id': [np.nan, np.nan], 'value': ['a', 'b']})
        df2 = pd.DataFrame({'id': [np.nan, np.nan], 'value': ['x', 'y']})
        
        analyze_join_keys(df1, df2, on='id')  # Should run without errors

    def test_duplicate_keys(self):
        """Test with duplicate keys"""
        df1 = pd.DataFrame({'id': [1, 1, 2, 2], 'value': ['a', 'b', 'c', 'd']})
        df2 = pd.DataFrame({'id': [1, 1, 3, 3], 'value': ['x', 'y', 'z', 'w']})
        
        analyze_join_keys(df1, df2, on='id')  # Should detect many-to-many relationship

    def test_calculate_join_sizes(self):
        """Test _calculate_join_sizes helper function"""
        from frameon.api.utils import _calculate_join_sizes
        
        df1 = pd.DataFrame({'id': [1, 1, 2, 2]})
        df2 = pd.DataFrame({'id': [1, 2, 2, 3]})
        
        result = _calculate_join_sizes(df1, df2, 'id', 'id')
        assert result['inner'] == 6
        assert result['left'] == 6
        assert result['right'] == 7
        assert result['outer'] == 7

    def test_resolve_keys(self):
        """Test _resolve_keys helper function"""
        from frameon.api.utils import _resolve_keys
        
        # Test single key
        result = _resolve_keys(on='id')
        assert result == {'left': ['id'], 'right': ['id']}
        
        # Test composite keys
        result = _resolve_keys(left_on=['id1', 'id2'], right_on=['key1', 'key2'])
        assert result == {'left': ['id1', 'id2'], 'right': ['key1', 'key2']}
    
    def test_composite_keys(self):
        """Test analysis with composite keys"""
        df1 = pd.DataFrame({
            'id1': [1, 2, 3],
            'id2': ['a', 'b', 'c'],
            'value': [10, 20, 30]
        })
        df2 = pd.DataFrame({
            'id1': [2, 3, 4],
            'id2': ['b', 'c', 'd'],
            'value': [200, 300, 400]
        })
        
        analyze_join_keys(df1, df2, left_on=['id1', 'id2'], right_on=['id1', 'id2'])
    
    @pytest.mark.parametrize("how", ['inner', 'left', 'right', 'outer'])
    def test_join_types(self, how):
        """Test different join types"""
        df1 = pd.DataFrame({'id': [1, 2, 3]})
        df2 = pd.DataFrame({'id': [2, 3, 4]})
        
        analyze_join_keys(df1, df2, on='id', how=how)

    def test_join_with_nan_keys(self):
        """Test join analysis with NaN values in keys"""
        df1 = pd.DataFrame({'id': [1, 2, np.nan], 'value': ['a', 'b', 'c']})
        df2 = pd.DataFrame({'id': [2, np.nan, 4], 'value': ['x', 'y', 'z']})
        analyze_join_keys(df1, df2, on='id')

    def test_mixed_type_keys(self):
        """Test with mixed type keys"""
        df1 = pd.DataFrame({'id': ['1', '2', '3'], 'value': ['a', 'b', 'c']})
        df2 = pd.DataFrame({'id': [2, 3, 4], 'value': ['x', 'y', 'z']})
        analyze_join_keys(df1, df2, left_on='id', right_on='id')

class TestFindInconsistentMappings:
    """Tests for find_inconsistent_mappings function"""
    
    def test_consistent_mappings(self):
        """Test with consistent one-to-one mappings"""
        df = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'value': ['a', 'a', 'b', 'b', 'c', 'c']
        })
        
        result = find_inconsistent_mappings(df, 'id', 'value', verbose=False)
        assert result.empty, "Should find no inconsistencies"
    
    def test_inconsistent_mappings(self):
        """Test detection of inconsistent mappings"""
        df = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'value': ['a', 'b', 'b', 'b', 'c', 'd']
        })
        
        result = find_inconsistent_mappings(df, 'id', 'value', verbose=False)
        assert len(result) == 4, "Should find two inconsistent keys"
        assert set(result['id']) == {1, 3}

    def test_categorical_mappings(self):
        """Test with categorical data"""
        df = pd.DataFrame({
            'id': pd.Categorical([1, 1, 2, 2, 3, 3]),
            'value': pd.Categorical(['a', 'b', 'b', 'b', 'c', 'd'])
        })
        result = find_inconsistent_mappings(df, 'id', 'value', verbose=False)
        assert len(result) == 4

    def test_empty_dataframe(self):
        """Test with empty DataFrame"""
        df = pd.DataFrame(columns=['id', 'value'])
        
        with pytest.warns(UserWarning, match="No inconsistent mappings found"):
            result = find_inconsistent_mappings(df, 'id', 'value', verbose=False)
        assert result.empty

    def test_no_inconsistencies(self):
        """Test when no inconsistencies exist"""
        df = pd.DataFrame({
            'id': [1, 1, 2, 2],
            'value': ['a', 'a', 'b', 'b']
        })
        
        with pytest.warns(UserWarning, match="No inconsistent mappings found"):
            result = find_inconsistent_mappings(df, 'id', 'value', verbose=False)
        assert result.empty

    def test_return_summary_mode(self):
        """Test return_summary=True mode"""
        df = pd.DataFrame({
            'id': [1, 1, 2, 2, 3, 3],
            'value': ['a', 'b', 'b', 'b', 'c', 'd']
        })
        
        result = find_inconsistent_mappings(df, 'id', 'value', return_summary=True, verbose=False)
        assert isinstance(result, dict)
        assert result['inconsistent_keys_count'] == 2
        assert result['affected_rows'] == 4
        assert pytest.approx(result['inconsistency_rate']) == 2/3

    def test_format_inconsistency_summary(self):
        """Test _format_inconsistency_summary helper"""
        from frameon.api.utils import _format_inconsistency_summary
        
        test_data = {
            'inconsistent_keys_count': 2,
            'affected_rows': 4,
            'inconsistency_rate': 0.5,
            'value_distribution': {2: 1, 3: 1},
            'most_inconsistent_keys': {
                1: {'unique_values': 2, 'total_rows': 2, 'sample_values': ['a', 'b']},
                3: {'unique_values': 2, 'total_rows': 2, 'sample_values': ['c', 'd']}
            }
        }
        
        report = _format_inconsistency_summary(test_data, 'id', 'value')
        assert "Inconsistent keys: 2" in report
        assert "Affected rows: 4" in report
        assert "Inconsistency rate: 50.0%" in report

class TestHaversineVectorized:
    """Tests for haversine_vectorized function"""
    
    def test_single_pair(self):
        """Test distance calculation between two points"""
        lat1, lon1 = 40.7128, -74.0060  # New York
        lat2, lon2 = 34.0522, -118.2437  # Los Angeles
        
        distance = haversine_vectorized(lat1, lon1, lat2, lon2)
        assert pytest.approx(distance, rel=0.01) == 3935  # ~3935 km
        
    def test_multiple_points(self):
        """Test vectorized distance calculation"""
        lats1 = np.array([40.7128, 51.5074])  # NY, London
        lons1 = np.array([-74.0060, -0.1278])
        lats2 = np.array([34.0522, 48.8566])  # LA, Paris
        lons2 = np.array([-118.2437, 2.3522])
        
        distances = haversine_vectorized(lats1, lons1, lats2, lons2)
        assert len(distances) == 2
        assert pytest.approx(distances[0], rel=0.01) == 3935  # NY-LA
        assert pytest.approx(distances[1], rel=0.01) == 342  # London-Paris

    def test_invalid_coordinates(self):
        """Test with invalid coordinates"""
        with pytest.raises(ValueError):
            haversine_vectorized(100, -200, 90, 180)  # Invalid lat/lon

    def test_different_units(self):
        """Test distance calculation with different units"""
        lat1, lon1 = 40.7128, -74.0060  # New York
        lat2, lon2 = 34.0522, -118.2437  # Los Angeles
        
        # Test kilometers (default)
        distance_km = haversine_vectorized(lat1, lon1, lat2, lon2, unit='km')
        assert pytest.approx(distance_km, rel=0.01) == 3935
        
        # Test meters
        distance_m = haversine_vectorized(lat1, lon1, lat2, lon2, unit='m')
        assert pytest.approx(distance_m, rel=0.01) == 3935 * 1000
        
        # Test miles
        distance_mi = haversine_vectorized(lat1, lon1, lat2, lon2, unit='mi')
        assert pytest.approx(distance_mi, rel=0.01) == 3935 * 0.621371

    def test_invalid_unit(self):
        """Test with invalid unit parameter"""
        with pytest.raises(ValueError, match="Invalid unit"):
            haversine_vectorized(40.7, -74.0, 34.0, -118.2, unit='invalid')

    def test_nan_handling(self):
        """Test handling of NaN values in coordinates"""
        lat1 = np.array([40.7128, np.nan])  # NY, invalid
        lon1 = np.array([-74.0060, -0.1278])
        lat2 = np.array([34.0522, 48.8566])  # LA, Paris
        lon2 = np.array([-118.2437, 2.3522])
        
        distances = haversine_vectorized(lat1, lon1, lat2, lon2)
        assert len(distances) == 2
        assert pytest.approx(distances[0], rel=0.01) == 3935  # NY-LA
        assert np.isnan(distances[1])  # Invalid point

    def test_mixed_input_types(self):
        """Test with mixed scalar and array inputs"""
        # Scalar point to array of points
        lat1, lon1 = 40.7128, -74.0060  # New York
        lats2 = np.array([34.0522, 48.8566])  # LA, Paris
        lons2 = np.array([-118.2437, 2.3522])
        
        distances = haversine_vectorized(lat1, lon1, lats2, lons2)
        assert len(distances) == 2
        assert pytest.approx(distances[0], rel=0.01) == 3935  # NY-LA
        assert pytest.approx(distances[1], rel=0.01) == 5839  # NY-Paris

    def test_same_point(self):
        """Test distance calculation for same point"""
        distance = haversine_vectorized(40.7128, -74.0060, 40.7128, -74.0060)
        assert distance == 0.0

    def test_antipodal_points(self):
        """Test distance calculation for antipodal points"""
        distance = haversine_vectorized(0, 0, 0, 180)  # Opposite sides of Earth
        assert pytest.approx(distance, rel=0.01) == 20015  # Earth's circumference ~20015 km