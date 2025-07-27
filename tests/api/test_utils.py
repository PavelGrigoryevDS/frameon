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