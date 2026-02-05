#!/usr/bin/env python3
"""
Dimensionality Reduction Module
Implements PCA and related preprocessing for the volatility research project.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional
import pickle

from .utils import logger


class VolatilityPCA:
    """
    Principal Component Analysis for volatility features.
    
    Handles standardization, PCA transformation, and interpretation.
    """
    
    def __init__(self, variance_threshold: float = 0.95, 
                 min_components: int = 3,
                 max_components: int = 5):
        """
        Initialize PCA analyzer.
        
        Args:
            variance_threshold: Cumulative variance to retain (e.g., 0.95 = 95%)
            min_components: Minimum number of components to keep
            max_components: Maximum number of components to keep
        """
        self.variance_threshold = variance_threshold
        self.min_components = min_components
        self.max_components = max_components
        
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_names = None
        self.n_components = None
        
    def fit(self, X: pd.DataFrame, feature_names: Optional[List[str]] = None) -> 'VolatilityPCA':
        """
        Fit PCA on training data.
        
        Args:
            X: Feature matrix (samples × features)
            feature_names: List of feature names (optional, inferred from DataFrame)
            
        Returns:
            Self for method chaining
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_array = X.values
        else:
            self.feature_names = feature_names
            X_array = X
        
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]
        
        logger.info(f"🔧 Standardizing {X_array.shape[1]} features...")
        
        # Standardize features (z-score normalization)
        X_scaled = self.scaler.fit_transform(X_array)
        
        # Determine number of components
        # First, fit with all components to see variance explained
        pca_full = PCA()
        pca_full.fit(X_scaled)
        
        cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumsum_var >= self.variance_threshold) + 1
        
        # Enforce min/max constraints
        n_components = max(self.min_components, n_components)
        n_components = min(self.max_components, n_components)
        n_components = min(n_components, X_array.shape[1])  # Can't exceed n_features
        
        self.n_components = n_components
        
        logger.info(f"📊 Retaining {n_components} components "
                   f"(explains {cumsum_var[n_components-1]:.2%} of variance)")
        
        # Fit PCA with selected number of components
        self.pca = PCA(n_components=n_components)
        self.pca.fit(X_scaled)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted PCA.
        
        Args:
            X: Feature matrix to transform
            
        Returns:
            Principal component scores
        """
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Standardize and transform
        X_scaled = self.scaler.transform(X_array)
        X_pca = self.pca.transform(X_scaled)
        
        return X_pca
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def get_loadings(self) -> pd.DataFrame:
        """
        Get principal component loadings (feature contributions).
        
        Returns:
            DataFrame of loadings (features × components)
        """
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.n_components)],
            index=self.feature_names
        )
        
        return loadings
    
    def get_variance_explained(self) -> pd.DataFrame:
        """
        Get variance explained by each component.
        
        Returns:
            DataFrame with variance metrics
        """
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        variance_df = pd.DataFrame({
            'component': [f'PC{i+1}' for i in range(self.n_components)],
            'variance_explained': self.pca.explained_variance_,
            'variance_ratio': self.pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(self.pca.explained_variance_ratio_)
        })
        
        return variance_df
    
    def interpret_components(self, top_k: int = 5) -> dict:
        """
        Interpret each principal component by identifying top contributing features.
        
        Args:
            top_k: Number of top features to identify per component
            
        Returns:
            Dictionary mapping PC names to top features and their loadings
        """
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        loadings = self.get_loadings()
        interpretations = {}
        
        for pc in loadings.columns:
            # Get absolute loadings (magnitude of contribution)
            abs_loadings = loadings[pc].abs().sort_values(ascending=False)
            top_features = abs_loadings.head(top_k)
            
            # Get actual loadings (with sign)
            top_loadings = loadings.loc[top_features.index, pc]
            
            interpretations[pc] = {
                'top_features': list(top_features.index),
                'loadings': list(top_loadings.values),
                'abs_loadings': list(top_features.values)
            }
        
        return interpretations
    
    def save(self, filepath: str):
        """Save fitted PCA model to disk."""
        if self.pca is None:
            raise ValueError("PCA not fitted. Nothing to save.")
        
        model_dict = {
            'scaler': self.scaler,
            'pca': self.pca,
            'feature_names': self.feature_names,
            'n_components': self.n_components,
            'variance_threshold': self.variance_threshold
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_dict, f)
        
        logger.info(f"💾 PCA model saved: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'VolatilityPCA':
        """Load fitted PCA model from disk."""
        with open(filepath, 'rb') as f:
            model_dict = pickle.load(f)
        
        # Create instance and restore attributes
        instance = cls(variance_threshold=model_dict['variance_threshold'])
        instance.scaler = model_dict['scaler']
        instance.pca = model_dict['pca']
        instance.feature_names = model_dict['feature_names']
        instance.n_components = model_dict['n_components']
        
        logger.info(f"📂 PCA model loaded: {filepath}")
        return instance


def select_features_for_pca(df: pd.DataFrame, 
                            exclude_patterns: List[str] = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select appropriate features for PCA analysis.
    
    Args:
        df: Master features dataframe
        exclude_patterns: List of patterns to exclude from features
        
    Returns:
        Tuple of (selected features dataframe, list of feature names)
    """
    if exclude_patterns is None:
        exclude_patterns = ['date', '_target', '_lag']
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude patterns
    selected_features = []
    for col in numeric_cols:
        if not any(pattern in col for pattern in exclude_patterns):
            selected_features.append(col)
    
    logger.info(f"📋 Selected {len(selected_features)} features for PCA")
    logger.info(f"   Excluded patterns: {exclude_patterns}")
    
    # Remove features with too many missing values
    missing_pct = df[selected_features].isnull().sum() / len(df)
    valid_features = missing_pct[missing_pct < 0.5].index.tolist()
    
    if len(valid_features) < len(selected_features):
        dropped = set(selected_features) - set(valid_features)
        logger.warning(f"⚠️  Dropped {len(dropped)} features with >50% missing: {dropped}")
        selected_features = valid_features
    
    # Return feature matrix (drop NaN rows)
    X = df[selected_features].dropna()
    
    logger.info(f"✅ Final feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    
    return X, selected_features


def create_component_names(interpretations: dict) -> dict:
    """
    Create intuitive names for principal components based on loadings.
    
    Args:
        interpretations: Output from VolatilityPCA.interpret_components()
        
    Returns:
        Dictionary mapping PC names to descriptive names
    """
    component_names = {}
    
    for pc, info in interpretations.items():
        top_feature = info['top_features'][0]
        
        # Create descriptive name based on top feature
        if 'vol' in top_feature:
            component_names[pc] = f"{pc}: Volatility Factor"
        elif 'sent' in top_feature:
            component_names[pc] = f"{pc}: Sentiment Factor"
        elif 'return' in top_feature:
            component_names[pc] = f"{pc}: Return Factor"
        elif any(x in top_feature for x in ['cl', 'ng', 'hg']):
            component_names[pc] = f"{pc}: Commodity Factor"
        else:
            component_names[pc] = f"{pc}: Mixed Factor"
    
    return component_names
