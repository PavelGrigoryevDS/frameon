import pytest
import pandas as pd
import numpy as np
from frameon.dataframe.analysis.text_analysis import TextAnalyzer
from frameon.core.base import FrameOn

@pytest.fixture
def sample_text_data():
    """Fixture with sample text data for testing"""
    texts = [
        "This is great! I love it!",
        "Terrible experience, would not recommend",
        "It's okay, nothing special",
        "Amazing product, works perfectly",
        "Worst purchase ever"
    ]
    return FrameOn({
        'text': texts,
        'category': ['positive', 'negative', 'neutral', 'positive', 'negative']
    })

class TestTextAnalyzer:
    """Tests for TextAnalyzer class"""
    
    def test_sentiment_analysis(self, sample_text_data):
        """Test sentiment analysis with different methods"""
        analyzer = TextAnalyzer(sample_text_data)
        
        # Test vader method
        vader_result = analyzer.sentiment(
            text_column='text',
            method='vader'
        )
        assert vader_result is not None
        
        # Test textblob method
        textblob_result = analyzer.sentiment(
            text_column='text',
            method='textblob'
        )
        assert textblob_result is not None
        
    def test_word_frequency(self, sample_text_data):
        """Test word frequency analysis"""
        analyzer = TextAnalyzer(sample_text_data)
        
        # Test top words
        top_result = analyzer.word_frequency(
            text_column='text',
            show='top'
        )
        assert top_result is not None
        
        # Test bottom words
        bottom_result = analyzer.word_frequency(
            text_column='text',
            show='bottom'
        )
        assert bottom_result is not None
        
    def test_input_validation(self, sample_text_data):
        """Test input validation"""
        analyzer = TextAnalyzer(sample_text_data)
        
        # Test invalid method
        with pytest.raises(ValueError):
            analyzer.sentiment(
                text_column='text',
                method='invalid_method'
            )
            
        # Test invalid show parameter
        with pytest.raises(ValueError):
            analyzer.word_frequency(
                text_column='text',
                show='invalid'
            )

    def test_edge_cases(self):
        """Test edge cases"""
        # Test empty DataFrame
        analyzer = TextAnalyzer(FrameOn())
        with pytest.raises(ValueError):
            analyzer.sentiment(
                text_column='text'
            )
            
        # Test empty text
        df = FrameOn({
            'text': ['', '   ', None],
            'category': ['a', 'b', 'c']
        })
        analyzer = TextAnalyzer(df)
        result = analyzer.sentiment(
            text_column='text'
        )
        assert result is not None
        
    @pytest.mark.parametrize("method", ['vader', 'textblob'])
    def test_sentiment_methods(self, sample_text_data, method):
        """Test sentiment analysis with different methods"""
        analyzer = TextAnalyzer(sample_text_data)
        result = analyzer.sentiment(text_column='text', method=method)
        assert result is not None
            
    @pytest.mark.parametrize("show", ['top', 'bottom', 'both'])
    def test_word_frequency_show_options(self, sample_text_data, show):
        """Test word frequency analysis with different show options"""
        analyzer = TextAnalyzer(sample_text_data)
        result = analyzer.word_frequency(text_column='text', show=show)
        assert result is not None
            