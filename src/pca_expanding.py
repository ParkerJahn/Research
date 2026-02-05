#!/usr/bin/env python3
"""
Expanding-Window PCA Module (Requirements-Compliant)
=====================================================

Implements CRITICAL PCA requirements:
1. PCA must be expanding-window (not full sample)
2. Fit PCA only on training data
3. Freeze loadings when projecting forward
4. Retain PCs explaining 70-85% variance

FATAL ERROR if violated: PCA fit on the full dataset
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional, Dict
import pickle

from .utils import logger


class ExpandingWindowPCA:
    """
    Expanding-window PCA for leakage-safe factor extraction.
    
    Key principle: At time t, we only use data from 1..t-1 to fit PCA.
    This prevents look-ahead bias and makes results realistic.
    """
    
    def __init__(
        self,
        variance_threshold: float = 0.80,  # 70-85% per requirements
        min_components: int = 3,
        max_components: int = 5,
        min_observations: int = 126  # ~6 months minimum
    ):
        """
        Initialize expanding-window PCA.
        
        Args:
            variance_threshold: Target cumulative variance (0.70-0.85 per requirements)
            min_components: Minimum components to retain
            max_components: Maximum components to retain
            min_observations: Minimum observations before fitting PCA
        """
        if not 0.70 <= variance_threshold <= 0.85:
            logger.warning(f"⚠️  Variance threshold {variance_threshold} outside recommended 70-85% range")
        
        self.variance_threshold = variance_threshold
        self.min_components = min_components
        self.max_components = max_components
        self.min_observations = min_observations
        
        # State
        self.scalers = {}  # Window end date -> fitted scaler
        self.pca_models = {}  # Window end date -> fitted PCA
        self.feature_names = None
        self.current_scaler = None
        self.current_pca = None
        
    def fit_transform_expanding(
        self,
        data: pd.DataFrame,
        date_col: str = 'date',
        refit_frequency: int = 20  # Refit every 20 days
    ) -> pd.DataFrame:
        """
        Fit PCA using expanding window and transform data.
        
        At each time t, fits PCA on data from start to t-1,
        then projects observation at t.
        
        Args:
            data: DataFrame with features
            date_col: Name of date column
            refit_frequency: Days between PCA refits (for efficiency)
            
        Returns:
            DataFrame with principal component scores
        """
        logger.info("="*60)
        logger.info("EXPANDING-WINDOW PCA")
        logger.info("="*60)
        
        # Separate date and features
        dates = data[date_col].values
        feature_cols = [c for c in data.columns if c != date_col]
        self.feature_names = feature_cols
        X = data[feature_cols].values
        
        n_obs = len(X)
        logger.info(f"📊 Data: {n_obs} observations, {len(feature_cols)} features")
        logger.info(f"📊 Refit frequency: every {refit_frequency} days")
        logger.info(f"📊 Variance threshold: {self.variance_threshold:.0%}")
        
        # Initialize output
        n_components = self.max_components
        pc_scores = np.full((n_obs, n_components), np.nan)
        
        # Track PCA fits
        fit_dates = []
        variance_explained = []
        
        last_fit_idx = -refit_frequency  # Force fit on first valid observation
        
        for i in range(self.min_observations, n_obs):
            # Check if we need to refit
            if i - last_fit_idx >= refit_frequency:
                # Fit on data up to (but not including) current observation
                X_train = X[:i]
                
                # Standardize
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_train)
                
                # Determine components
                pca_full = PCA()
                pca_full.fit(X_scaled)
                
                cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
                n_comp = np.argmax(cumsum_var >= self.variance_threshold) + 1
                n_comp = max(self.min_components, min(self.max_components, n_comp))
                
                # Fit final PCA
                pca = PCA(n_components=n_comp)
                pca.fit(X_scaled)
                
                # Store state
                self.current_scaler = scaler
                self.current_pca = pca
                self.scalers[dates[i]] = scaler
                self.pca_models[dates[i]] = pca
                
                last_fit_idx = i
                fit_dates.append(dates[i])
                variance_explained.append(cumsum_var[n_comp-1])
                
                if len(fit_dates) <= 3 or len(fit_dates) % 10 == 0:
                    logger.info(f"   📅 {dates[i]}: Fit {n_comp} PCs, {cumsum_var[n_comp-1]:.1%} variance")
            
            # Project current observation using frozen PCA
            X_curr = X[i:i+1]
            X_curr_scaled = self.current_scaler.transform(X_curr)
            pc_curr = self.current_pca.transform(X_curr_scaled)
            
            # Store scores (pad with NaN if fewer components)
            pc_scores[i, :pc_curr.shape[1]] = pc_curr[0]
        
        # Create output DataFrame
        pc_cols = [f'PC{j+1}' for j in range(n_components)]
        result = pd.DataFrame(pc_scores, columns=pc_cols)
        result[date_col] = dates
        
        # Reorder columns
        result = result[[date_col] + pc_cols]
        
        # Summary
        logger.info("\n" + "-"*60)
        logger.info("EXPANDING-WINDOW PCA SUMMARY")
        logger.info("-"*60)
        logger.info(f"   Total refits: {len(fit_dates)}")
        logger.info(f"   Average variance explained: {np.mean(variance_explained):.1%}")
        logger.info(f"   Output: {result.dropna().shape[0]} observations with PCA scores")
        
        return result
    
    def get_loadings(self, window_date: Optional = None) -> pd.DataFrame:
        """
        Get PCA loadings for interpretation.
        
        Args:
            window_date: Specific date's PCA (default: latest)
            
        Returns:
            DataFrame of loadings (features × components)
        """
        if window_date is None:
            pca = self.current_pca
        else:
            if window_date not in self.pca_models:
                raise ValueError(f"No PCA model for date {window_date}")
            pca = self.pca_models[window_date]
        
        if pca is None:
            raise ValueError("No PCA model fitted")
        
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)],
            index=self.feature_names
        )
        
        return loadings
    
    def interpret_components(self, top_k: int = 5) -> Dict:
        """
        Interpret each PC by top contributing features.
        
        Per Requirements.md expected interpretation:
        | PC  | Interpretation             |
        | --- | -------------------------- |
        | PC1 | Semiconductor equity level |
        | PC2 | Market volatility          |
        | PC3 | Sentiment shock            |
        | PC4 | Energy / commodities       |
        | PC5 | Cross-asset stress         |
        """
        loadings = self.get_loadings()
        interpretations = {}
        
        for pc in loadings.columns:
            abs_loadings = loadings[pc].abs().sort_values(ascending=False)
            top_features = abs_loadings.head(top_k)
            
            interpretations[pc] = {
                'top_features': list(top_features.index),
                'loadings': list(loadings.loc[top_features.index, pc].values),
                'interpretation': self._guess_interpretation(top_features.index.tolist())
            }
        
        return interpretations
    
    def _guess_interpretation(self, features: List[str]) -> str:
        """Guess PC interpretation from top features."""
        features_lower = ' '.join(features).lower()
        
        if any(x in features_lower for x in ['nvda', 'amd', 'intc', 'smh', 'soxx']):
            return "Semiconductor equity"
        elif any(x in features_lower for x in ['vol', 'vix', 'vvix']):
            return "Market volatility"
        elif any(x in features_lower for x in ['sent', 'shock']):
            return "Sentiment shock"
        elif any(x in features_lower for x in ['oil', 'gas', 'crude', 'cl', 'ng', 'bz']):
            return "Energy / commodities"
        elif any(x in features_lower for x in ['gold', 'silver', 'copper', 'gc', 'hg']):
            return "Cross-asset / metals"
        else:
            return "Mixed factor"
    
    def save(self, filepath: str):
        """Save PCA state."""
        state = {
            'variance_threshold': self.variance_threshold,
            'min_components': self.min_components,
            'max_components': self.max_components,
            'min_observations': self.min_observations,
            'feature_names': self.feature_names,
            'current_scaler': self.current_scaler,
            'current_pca': self.current_pca,
            'scalers': self.scalers,
            'pca_models': self.pca_models
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"💾 Saved PCA state: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ExpandingWindowPCA':
        """Load PCA state."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        instance = cls(
            variance_threshold=state['variance_threshold'],
            min_components=state['min_components'],
            max_components=state['max_components'],
            min_observations=state['min_observations']
        )
        instance.feature_names = state['feature_names']
        instance.current_scaler = state['current_scaler']
        instance.current_pca = state['current_pca']
        instance.scalers = state['scalers']
        instance.pca_models = state['pca_models']
        
        logger.info(f"📂 Loaded PCA state: {filepath}")
        return instance


def create_pca_features_leakage_safe(
    features_df: pd.DataFrame,
    sentiment_shock_df: Optional[pd.DataFrame] = None,
    date_col: str = 'date',
    variance_threshold: float = 0.80,
    refit_frequency: int = 20
) -> Tuple[pd.DataFrame, ExpandingWindowPCA]:
    """
    Create PCA features with proper leakage controls.
    
    Implements Requirements.md:
    1. PCA must be expanding-window
    2. Fit PCA only on training data
    3. Freeze loadings when projecting forward
    4. Retain PCs explaining 70-85% variance
    
    Args:
        features_df: DataFrame with raw features
        sentiment_shock_df: Optional pre-processed sentiment shocks
        date_col: Name of date column
        variance_threshold: Target variance (0.70-0.85)
        refit_frequency: Days between refits
        
    Returns:
        Tuple of (PCA scores DataFrame, fitted PCA object)
    """
    logger.info("Creating PCA features (leakage-safe)...")
    
    # Combine features
    if sentiment_shock_df is not None:
        # Only use lagged sentiment shocks (never contemporaneous)
        shock_cols = [c for c in sentiment_shock_df.columns 
                     if 'lag' in c and 'shock' in c]
        
        if shock_cols:
            features_df = features_df.merge(
                sentiment_shock_df[[date_col] + shock_cols],
                on=date_col,
                how='left'
            )
            logger.info(f"   Added lagged sentiment shocks: {shock_cols}")
    
    # Drop rows with NaN
    features_df = features_df.dropna()
    
    # Initialize and fit expanding-window PCA
    pca = ExpandingWindowPCA(
        variance_threshold=variance_threshold,
        min_components=3,
        max_components=5
    )
    
    pc_scores = pca.fit_transform_expanding(
        features_df,
        date_col=date_col,
        refit_frequency=refit_frequency
    )
    
    # Print interpretation
    logger.info("\n" + "-"*60)
    logger.info("PCA INTERPRETATION")
    logger.info("-"*60)
    
    interp = pca.interpret_components()
    for pc, info in interp.items():
        logger.info(f"   {pc}: {info['interpretation']}")
        for feat, load in zip(info['top_features'][:3], info['loadings'][:3]):
            logger.info(f"      {load:+.3f} {feat}")
    
    return pc_scores, pca
