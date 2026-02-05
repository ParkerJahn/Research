#!/usr/bin/env python3
"""
Statistical Models Module
Implements VAR, Granger causality, and forecasting for volatility research.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
import pickle
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tools.eval_measures import rmse, meanabs
import warnings

from .utils import logger


class VolatilityVAR:
    """
    Vector Autoregression model for volatility prediction.
    
    Handles model estimation, diagnostics, Granger causality, and forecasting.
    """
    
    def __init__(self, maxlags: int = 10, ic: str = 'aic'):
        """
        Initialize VAR model.
        
        Args:
            maxlags: Maximum number of lags to test
            ic: Information criterion ('aic' or 'bic')
        """
        self.maxlags = maxlags
        self.ic = ic
        self.model = None
        self.results = None
        self.selected_lag = None
        self.endog_names = None
        self.exog_names = None
        
    def test_stationarity(self, data: pd.DataFrame, 
                         variables: List[str]) -> pd.DataFrame:
        """
        Test stationarity using Augmented Dickey-Fuller test.
        
        Args:
            data: DataFrame with time series
            variables: List of variable names to test
            
        Returns:
            DataFrame with test results
        """
        results = []
        
        for var in variables:
            if var not in data.columns:
                logger.warning(f"⚠️  Variable {var} not found in data")
                continue
            
            series = data[var].dropna()
            
            if len(series) < 10:
                logger.warning(f"⚠️  {var}: Too few observations ({len(series)})")
                continue
            
            try:
                adf_result = adfuller(series, autolag='AIC')
                
                results.append({
                    'variable': var,
                    'adf_statistic': adf_result[0],
                    'p_value': adf_result[1],
                    'n_lags_used': adf_result[2],
                    'n_observations': adf_result[3],
                    'is_stationary': adf_result[1] < 0.05,
                    'critical_1pct': adf_result[4]['1%'],
                    'critical_5pct': adf_result[4]['5%'],
                    'critical_10pct': adf_result[4]['10%']
                })
            except Exception as e:
                logger.error(f"❌ Error testing {var}: {e}")
        
        return pd.DataFrame(results)
    
    def select_lag_order(self, endog_data: pd.DataFrame, 
                        exog_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Select optimal lag order using information criteria.
        
        Args:
            endog_data: Endogenous variables
            exog_data: Exogenous variables (optional)
            
        Returns:
            Dictionary with lag selection results
        """
        logger.info(f"🔍 Testing lag orders 1 to {self.maxlags}...")
        
        # Fit VAR model to get lag order selection
        model = VAR(endog_data, exog=exog_data)
        
        # Get lag order selection criteria
        lag_order_results = model.select_order(maxlags=self.maxlags)
        
        selected_lag = lag_order_results.selected_orders[self.ic]
        
        logger.info(f"✅ Selected lag order ({self.ic.upper()}): {selected_lag}")
        
        return {
            'selected_lag': selected_lag,
            'aic': lag_order_results.aic,
            'bic': lag_order_results.bic,
            'fpe': lag_order_results.fpe,
            'hqic': lag_order_results.hqic
        }
    
    def fit(self, endog_data: pd.DataFrame, 
            exog_data: Optional[pd.DataFrame] = None,
            lags: Optional[int] = None) -> 'VolatilityVAR':
        """
        Fit VAR model.
        
        Args:
            endog_data: Endogenous variables (to be predicted)
            exog_data: Exogenous variables (predictors)
            lags: Number of lags (if None, will be selected automatically)
            
        Returns:
            Self for method chaining
        """
        self.endog_names = list(endog_data.columns)
        if exog_data is not None:
            self.exog_names = list(exog_data.columns)
        
        # Select lag order if not provided
        if lags is None:
            lag_results = self.select_lag_order(endog_data, exog_data)
            lags = lag_results['selected_lag']
        
        self.selected_lag = lags
        
        logger.info(f"🔧 Fitting VAR model with {lags} lags...")
        logger.info(f"   Endogenous variables: {self.endog_names}")
        if self.exog_names:
            logger.info(f"   Exogenous variables: {self.exog_names}")
        
        # Fit model
        self.model = VAR(endog_data, exog=exog_data)
        self.results = self.model.fit(maxlags=lags, ic=self.ic)
        
        logger.info(f"✅ VAR model fitted successfully")
        
        return self
    
    def get_coefficients(self) -> pd.DataFrame:
        """
        Extract model coefficients in readable format.
        
        Returns:
            DataFrame with coefficients, standard errors, t-stats, p-values
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        coef_data = []
        
        for i, eq_name in enumerate(self.endog_names):
            params = self.results.params.iloc[:, i]
            stderr = self.results.stderr.iloc[:, i]
            tvalues = self.results.tvalues.iloc[:, i]
            pvalues = self.results.pvalues.iloc[:, i]
            
            for var_name, coef, se, tval, pval in zip(
                params.index, params.values, stderr.values, 
                tvalues.values, pvalues.values
            ):
                coef_data.append({
                    'equation': eq_name,
                    'variable': var_name,
                    'coefficient': coef,
                    'std_error': se,
                    't_statistic': tval,
                    'p_value': pval,
                    'significant': pval < 0.05
                })
        
        return pd.DataFrame(coef_data)
    
    def run_diagnostics(self) -> Dict:
        """
        Run model diagnostics tests.
        
        Returns:
            Dictionary with diagnostic test results
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        logger.info("🔬 Running model diagnostics...")
        
        diagnostics = {}
        
        # Residual tests
        try:
            # Portmanteau test (residual autocorrelation)
            portmanteau = self.results.test_whiteness(nlags=10)
            # Handle different statsmodels versions (attribute may be .test_statistic, .statistic, or .lm)
            stat = getattr(portmanteau, 'test_statistic', None) or getattr(portmanteau, 'lm', None)
            diagnostics['portmanteau_statistic'] = stat
            diagnostics['portmanteau_pvalue'] = portmanteau.pvalue
            diagnostics['portmanteau_conclusion'] = (
                "No autocorrelation" if portmanteau.pvalue > 0.05 
                else "Autocorrelation detected"
            )
        except Exception as e:
            logger.warning(f"⚠️  Portmanteau test failed: {e}")
            diagnostics['portmanteau_statistic'] = None
        
        # Normality test
        try:
            normality = self.results.test_normality()
            # Handle different statsmodels versions (attribute may be .test_statistic, .statistic, or .joint_statistic)
            stat = getattr(normality, 'test_statistic', None) or getattr(normality, 'joint_statistic', None)
            diagnostics['normality_statistic'] = stat
            diagnostics['normality_pvalue'] = normality.pvalue
            diagnostics['normality_conclusion'] = (
                "Residuals normal" if normality.pvalue > 0.05
                else "Residuals non-normal"
            )
        except Exception as e:
            logger.warning(f"⚠️  Normality test failed: {e}")
            diagnostics['normality_statistic'] = None
        
        # Model fit statistics
        diagnostics['aic'] = self.results.aic
        diagnostics['bic'] = self.results.bic
        diagnostics['fpe'] = self.results.fpe
        diagnostics['hqic'] = self.results.hqic
        diagnostics['log_likelihood'] = self.results.llf
        
        return diagnostics
    
    def granger_causality(self, exog_var: str, endog_var: str,
                         maxlag: Optional[int] = None) -> pd.DataFrame:
        """
        Test Granger causality: does exog_var Granger-cause endog_var?
        
        Args:
            exog_var: Exogenous variable (cause)
            endog_var: Endogenous variable (effect)
            maxlag: Maximum lag to test (defaults to model lag)
            
        Returns:
            DataFrame with test results for each lag
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if maxlag is None:
            maxlag = self.selected_lag
        
        # Granger causality requires the full dataset
        # We'll use the original data used to fit the model
        endog_data = self.results.endog
        
        # Create DataFrame for testing
        data = pd.DataFrame(endog_data, columns=self.endog_names)
        
        # If testing exogenous -> endogenous, we need to add exog to data
        if exog_var not in data.columns:
            logger.warning(f"⚠️  {exog_var} not in endogenous variables")
            logger.warning("    Granger causality tests only work within VAR system")
            return pd.DataFrame()
        
        try:
            # Run Granger causality test
            # Note: grangercausalitytests tests if column 0 Granger-causes column 1
            test_data = data[[exog_var, endog_var]]
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gc_results = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)
            
            # Extract results
            results = []
            for lag in range(1, maxlag + 1):
                # Get F-test results (ssr_ftest)
                ftest = gc_results[lag][0]['ssr_ftest']
                
                results.append({
                    'cause': exog_var,
                    'effect': endog_var,
                    'lag': lag,
                    'f_statistic': ftest[0],
                    'p_value': ftest[1],
                    'significant': ftest[1] < 0.05
                })
            
            return pd.DataFrame(results)
        
        except Exception as e:
            logger.error(f"❌ Granger causality test failed: {e}")
            return pd.DataFrame()
    
    def impulse_response(self, periods: int = 10, 
                        orthogonalized: bool = True) -> np.ndarray:
        """
        Compute impulse response functions.
        
        Args:
            periods: Number of periods ahead
            orthogonalized: Whether to orthogonalize shocks
            
        Returns:
            Array of IRF values (periods × equations × equations)
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        logger.info(f"📈 Computing impulse responses for {periods} periods...")
        
        if orthogonalized:
            irf = self.results.irf(periods)
        else:
            irf = self.results.irf(periods)
        
        return irf
    
    def forecast(self, steps: int = 1, 
                exog_future: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps ahead to forecast
            exog_future: Future values of exogenous variables
            
        Returns:
            DataFrame with forecasts
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get forecast
        forecast = self.results.forecast(
            self.results.endog[-self.selected_lag:],
            steps=steps,
            exog_future=exog_future
        )
        
        # Create DataFrame
        forecast_df = pd.DataFrame(
            forecast,
            columns=self.endog_names
        )
        
        return forecast_df
    
    def forecast_intervals(self, steps: int = 1, alpha: float = 0.05,
                          exog_future: Optional[np.ndarray] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate forecasts with confidence intervals.
        
        Args:
            steps: Number of steps ahead
            alpha: Significance level (0.05 = 95% CI)
            exog_future: Future exogenous values
            
        Returns:
            Tuple of (point forecast, lower bound, upper bound)
        """
        if self.results is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get forecast with intervals
        forecast = self.results.forecast_interval(
            self.results.endog[-self.selected_lag:],
            steps=steps,
            alpha=alpha,
            exog_future=exog_future
        )
        
        point_forecast = pd.DataFrame(forecast[0], columns=self.endog_names)
        lower_bound = pd.DataFrame(forecast[1], columns=self.endog_names)
        upper_bound = pd.DataFrame(forecast[2], columns=self.endog_names)
        
        return point_forecast, lower_bound, upper_bound
    
    def save(self, filepath: str):
        """Save fitted VAR model."""
        if self.results is None:
            raise ValueError("Model not fitted. Nothing to save.")
        
        model_dict = {
            'results': self.results,
            'maxlags': self.maxlags,
            'ic': self.ic,
            'selected_lag': self.selected_lag,
            'endog_names': self.endog_names,
            'exog_names': self.exog_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_dict, f)
        
        logger.info(f"💾 VAR model saved: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'VolatilityVAR':
        """Load fitted VAR model."""
        with open(filepath, 'rb') as f:
            model_dict = pickle.load(f)
        
        instance = cls(maxlags=model_dict['maxlags'], ic=model_dict['ic'])
        instance.results = model_dict['results']
        instance.selected_lag = model_dict['selected_lag']
        instance.endog_names = model_dict['endog_names']
        instance.exog_names = model_dict['exog_names']
        
        logger.info(f"📂 VAR model loaded: {filepath}")
        return instance


def calculate_forecast_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """
    Calculate forecast accuracy metrics.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()
    
    # Remove NaN pairs
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) == 0:
        return {}
    
    # Calculate metrics
    metrics = {
        'rmse': rmse(actual, predicted),
        'mae': meanabs(actual, predicted),
        'mape': np.mean(np.abs((actual - predicted) / actual)) * 100,
        'mse': np.mean((actual - predicted) ** 2),
        'directional_accuracy': np.mean(np.sign(actual[1:] - actual[:-1]) == 
                                       np.sign(predicted[1:] - predicted[:-1])) * 100
    }
    
    return metrics
