"""
Output Formatter: Rankings, tiers, strategy layers (Section 8).
Produces final deliverable: ranked molecules with investment recommendations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


class OutputFormatter:
    """
    Format ensemble results into:
    - Ranked molecule list
    - Investment tiers (High Potential / Moderate / Avoid)
    - Risk vs Opportunity breakdown
    - Strategy recommendations per molecule
    """
    
    # ===== INVESTMENT TIERS (Section 8.2) =====
    TIER_THRESHOLDS = {
        'HIGH_POTENTIAL': 70,
        'MODERATE': 50,
        'AVOID': 0
    }
    
    # ===== ENTRY STRATEGIES (Section 8.4) =====
    STRATEGIES = {
        'COST_LEADERSHIP': {
            'condition': lambda row: row['risk'] > 50 and row['market_size'] > 500,
            'description': 'Cost leadership: compete on price, large market'
        },
        'TENDER_STRATEGY': {
            'condition': lambda row: row['sector'] == 'HOSPITAL' and row['competition_count'] < 5,
            'description': 'Tender-based: focused on hospital sector with limited competition'
        },
        'PORTFOLIO_BUNDLING': {
            'condition': lambda row: row['market_penetration'] > 50 and row['price_change'] < -5,
            'description': 'Portfolio bundling: stable market with established presence'
        },
        'DIFFERENTIATION': {
            'condition': lambda row: row['revenue_growth'] > 30 and row['price_change'] > 5,
            'description': 'Differentiation: high growth + pricing power'
        },
        'MONOPOLY_DEFENSE': {
            'condition': lambda row: row.get('market_share', 0) > 80,
            'description': 'Monopoly defense: preserve share with pricing/contract management'
        },
        'NICHE_PARTNERSHIP': {
            'condition': lambda row: row.get('market_size', 0) < 10 and row.get('revenue_growth', 0) > 10,
            'description': 'Niche partnership: focus on specialist access and distribution'
        }
    }
    
    def __init__(self, ensemble_df: pd.DataFrame, features_df: pd.DataFrame):
        """
        Args:
            ensemble_df: From EnsembleDecisionEngine.get_ensemble_dataframe()
            features_df: From FeatureEngineer.build_features()
        """
        self.ensemble = ensemble_df.copy()
        self.features = features_df.copy()
        self.output = pd.DataFrame()
    
    # ===== CLASSIFICATION & RANKING =====
    
    def classify_investment_tier(self) -> pd.Series:
        """Classify molecules into tiers based on ensemble score."""
        ensemble_score = self.ensemble['ensemble_score']
        
        # Use simple logic instead of pd.cut to avoid bin edge issues
        tiers = pd.Series('Moderate', index=ensemble_score.index)
        tiers[ensemble_score >= self.TIER_THRESHOLDS['HIGH_POTENTIAL']] = 'High Potential'
        tiers[ensemble_score < self.TIER_THRESHOLDS['MODERATE']] = 'Avoid'
        
        return tiers
    
    def rank_molecules(self) -> Tuple[pd.DataFrame, int]:
        """
        Rank molecules by ensemble score (descending).
        Return ranked list + percentile rankings.
        """
        ranked = self.ensemble.copy()
        ranked['rank'] = ranked['ensemble_score'].rank(ascending=False, method='first')  # Use 'first' to avoid ties
        ranked['percentile'] = ranked['ensemble_score'].rank(pct=True) * 100
        
        # Sort by rank
        ranked = ranked.sort_values('rank').reset_index(drop=True)
        
        total_molecules = len(ranked)
        
        return ranked, total_molecules
    
    # ===== INTERPRETATION LAYER (Section 8.3) =====
    
    def extract_key_drivers(self, row: pd.Series) -> List[str]:
        """
        FIX #22: Add explainability - extract feature contributions to score.
        Identify key drivers of score for each molecule using importance ranking.
        """
        drivers = []
        
        rg = row.get('revenue_growth', 0)
        ms = row.get('market_size', 0)
        cc = row.get('competition_count', 0)
        pc = row.get('price_change', 0)
        ms_share = row.get('market_share', 0)
        vol = row.get('volatility', 0)
        opp_score = row.get('opportunity_score', 50)
        risk_score = row.get('risk_score', 50)

        # FIX #22: Calculate feature contributions (simple linear importance)
        # Features with highest absolute deviation from neutral contribute most
        contributions = {
            'growth': abs(rg - 0) if rg != 0 else 0,
            'market_size': abs(ms - 100) if ms != 100 else 0,
            'competition': abs(cc - 5) if cc != 5 else 0,
            'pricing': abs(pc - 0) if pc != 0 else 0,
            'concentration': abs(ms_share - 50) if ms_share != 50 else 0,
            'volatility': abs(vol - 0.1) if vol != 0.1 else 0,
        }
        
        # Sort by contribution magnitude (importance)
        sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        
        # Build driver statements with contribution weights
        if rg > 50:
            drivers.append(f"🔥 Strong growth ({rg:.0f}%)")
        elif rg < -10:
            drivers.append(f"⚠️ Declining demand ({rg:.0f}%)")
        else:
            drivers.append(f"📊 Stable growth ({rg:.0f}%)")

        if ms > 500:
            drivers.append(f"💰 Large market (${ms:.0f}M)")
        elif ms < 10:
            drivers.append(f"🎯 Niche market (${ms:.1f}M)")
        else:
            drivers.append(f"📈 Mid-size market (${ms:.1f}M)")

        drivers.append(f"🏢 Share: {ms_share:.1f}%")

        if cc > 15:
            drivers.append(f"🔀 Fragmented ({cc:.0f} competitors)")
        elif cc < 3:
            drivers.append(f"🛡️ Concentrated ({cc:.0f} competitors)")
        else:
            drivers.append(f"⚔️ Competitive ({cc:.0f} competitors)")

        if pc > 10:
            drivers.append(f"💵 Pricing power (+{pc:.1f}%)")
        elif pc < -10:
            drivers.append(f"📉 Price pressure ({pc:.1f}%)")

        if vol > 0.5:
            drivers.append("⚡ High volatility")
        
        return drivers if drivers else ["📊 Market characteristics unclear"]
    
    def extract_risks(self, row: pd.Series) -> List[str]:
        """Identify key risks for each molecule."""
        risks = []
        
        ms_share = row.get('market_share', 0)
        rg = row.get('revenue_growth', 0)
        ms = row.get('market_size', 0)
        vol = row.get('volatility', 0)
        gex = row.get('generic_erosion_index', 100)
        risk_score = row.get('risk_score', 0)

        if ms_share > 80:
            risks.append(f"Monopoly concentration ({ms_share:.1f}% share) increases regulatory and pricing exposure")
        elif risk_score > 70:
            risks.append(f"High competitive risk (score: {risk_score:.0f})")

        if vol > 0.5:
            risks.append("High volatility: unstable MAT trajectory across periods")

        if gex < 30:
            risks.append("Generic erosion pressure: price decline with volume expansion")

        if rg < -20:
            risks.append(f"Market decline: losing {abs(rg):.0f}% revenue")

        if ms < 1:
            risks.append("Limited market size: constrained opportunity and thin liquidity")
        
        if not risks:
            risks.append("None identified")
        
        return risks
    
    def extract_market_position(self, row: pd.Series) -> Dict:
        """Summarize market position for each molecule."""
        return {
            'market_share': round(row['market_share'], 1),
            'market_size': round(row['market_size'], 1),
            'sector': row['sector'],
            'country': row['country'],
            'manufacturer': row['manufacturer'],
            'lifecycle_stage': row.get('lifecycle_stage', 'Unknown')
        }
    
    # ===== STRATEGY RECOMMENDATIONS (Section 8.4) =====
    
    def recommend_entry_strategy(self, row: pd.Series) -> str:
        """Recommend entry strategy based on molecule characteristics."""
        # Priority-based, mutually exclusive assignment
        risk = row.get('risk_score', 0)
        market_share = row.get('market_share', 0)
        market_size = row.get('market_size', 0)
        growth = row.get('revenue_growth', 0)
        comp = row.get('competition_count', 0)
        sector = row.get('sector', '')
        price_change = row.get('price_change', 0)
        penetration = row.get('market_penetration', 0)

        if market_share > 80:
            return 'MONOPOLY_DEFENSE'
        if sector == 'HOSPITAL' and comp < 5:
            return 'TENDER_STRATEGY'
        if growth > 30 and price_change > 5:
            return 'DIFFERENTIATION'
        if market_size > 500 and risk > 50:
            return 'COST_LEADERSHIP'
        if market_size < 10 and growth > 10:
            return 'NICHE_PARTNERSHIP'
        if penetration > 50 and price_change < -5:
            return 'PORTFOLIO_BUNDLING'

        return 'MONITOR'
    
    # ===== PORTFOLIO STRATEGY (Section 8.5) =====
    
    def recommend_portfolio_balance(self, ranked_df: pd.DataFrame) -> Dict:
        """
        Portfolio strategy: balance high-risk/high-return vs stable.
        """
        high_potential = ranked_df[ranked_df['ensemble_score'] > 70]
        moderate = ranked_df[(ranked_df['ensemble_score'] >= 50) & (ranked_df['ensemble_score'] <= 70)]
        
        growth_col = 'revenue_growth_%' if 'revenue_growth_%' in ranked_df.columns else 'revenue_growth'
        high_growth = high_potential[high_potential[growth_col] > 30]
        stable = high_potential[high_potential[growth_col] <= 30]
        
        portfolio = {
            'high_growth_count': len(high_growth),
            'high_growth_examples': high_growth['molecule'].head(3).tolist(),
            'stable_count': len(stable),
            'stable_examples': stable['molecule'].head(3).tolist(),
            'moderate_count': len(moderate),
            'recommendation': (
                f"Recommended portfolio: {len(high_growth)} high-growth molecules + "
                f"{len(stable)} stable performers + {min(3, len(moderate))} moderate opportunities"
            )
        }
        
        return portfolio
    
    # ===== BUILD FINAL OUTPUT =====
    
    def build_output_report(self) -> pd.DataFrame:
        """Build comprehensive output DataFrame with all layers."""
        ranked, total = self.rank_molecules()
        
        output = pd.DataFrame(index=ranked.index)
        
        # Ranking
        output['rank'] = ranked['rank']
        output['percentile'] = ranked['percentile']
        output['ensemble_score'] = ranked['ensemble_score']
        output['investment_tier'] = self.classify_investment_tier()
        
        # Core metrics
        output['molecule_id'] = ranked['molecule_id']
        output['molecule'] = ranked['molecule']
        output['manufacturer'] = ranked['manufacturer']
        output['country'] = ranked['country']
        output['sector'] = ranked['sector']
        
        # Performance breakdown - use ensemble data and map back to features
        output['revenue_growth_%'] = output.index.map(
            lambda i: round(self.features.loc[i, 'revenue_growth'] if i in self.features.index else 0, 1)
        )
        output['revenue_growth'] = output['revenue_growth_%']
        output['market_size_$M'] = output.index.map(
            lambda i: round(self.features.loc[i, 'market_size'] if i in self.features.index else 0, 1)
        )
        output['competition_count'] = output.index.map(
            lambda i: int(self.features.loc[i, 'competition_count'] if i in self.features.index else 0)
        )
        output['market_share_%'] = output.index.map(
            lambda i: round(self.features.loc[i, 'market_share'] if i in self.features.index else 0, 1)
        )
        output['volatility'] = output.index.map(
            lambda i: round(self.features.loc[i, 'volatility'] if i in self.features.index else 0, 3)
        )
        
        # Risk & Opportunity
        output['opportunity_score'] = ranked['opportunity'].apply(lambda x: round(x, 1))
        output['risk_score'] = ranked['risk'].apply(lambda x: round(x, 1))
        output['confidence_level'] = ranked['ensemble_confidence_class']
        
        # ML Insights
        output['ml_classification'] = ranked['rf_classification']
        output['market_segment'] = ranked['cluster_segment']
        
        # Strategic layers
        output['key_drivers'] = output.index.map(
            lambda i: ' | '.join(self.extract_key_drivers(self.features.loc[i])) if i in self.features.index else 'N/A'
        )
        output['risks'] = output.index.map(
            lambda i: ' | '.join(self.extract_risks(self.features.loc[i])) if i in self.features.index else 'None identified'
        )
        output['entry_strategy'] = output.index.map(
            lambda i: self.recommend_entry_strategy(self.features.loc[i]) if i in self.features.index else 'STANDARD'
        )
        
        self.output = output
        return output
    
    def export_summary_json(self) -> Dict:
        """Export high-level summary as JSON."""
        if self.output.empty:
            self.build_output_report()
        
        high_potential = self.output[self.output['investment_tier'] == 'High Potential']
        moderate = self.output[self.output['investment_tier'] == 'Moderate']
        avoid = self.output[self.output['investment_tier'] == 'Avoid']
        
        summary = {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'total_molecules': len(self.output),
            'investment_tiers': {
                'high_potential': {
                    'count': len(high_potential),
                    'avg_score': high_potential['ensemble_score'].mean(),
                    'avg_growth': high_potential['revenue_growth_%'].mean(),
                    'top_5': high_potential[['rank', 'molecule', 'ensemble_score']].head(5).to_dict('records')
                },
                'moderate': {
                    'count': len(moderate),
                    'avg_score': moderate['ensemble_score'].mean()
                },
                'avoid': {
                    'count': len(avoid),
                    'avg_score': avoid['ensemble_score'].mean()
                }
            },
            'portfolio_recommendation': self.recommend_portfolio_balance(self.output),
            'confidence_levels': self.output['confidence_level'].value_counts().to_dict()
        }
        
        return summary


# Example usage
if __name__ == "__main__":
    from data_layer import IQVIADataLoader
    from feature_engineering import FeatureEngineer
    from scoring_engine import ScoringEngine
    from ml_models import MLModels
    from ensemble import EnsembleDecisionEngine
    
    base_dir = Path(__file__).resolve().parent
    loader = IQVIADataLoader(base_dir / 'IQVIA - Leah .xlsx')
    loader.load()
    loader.validate_structure()
    clean_df = loader.clean()
    
    engineer = FeatureEngineer(clean_df)
    features = engineer.build_features()
    
    scorer = ScoringEngine(features)
    scores_df = scorer.get_scores_dataframe()
    
    ml = MLModels(features)
    ml.train_all_models()
    ml_df = ml.get_ml_predictions_dataframe()
    
    ensemble = EnsembleDecisionEngine(scores_df, ml_df)
    ensemble_df = ensemble.get_ensemble_dataframe()
    
    formatter = OutputFormatter(ensemble_df, features)
    output = formatter.build_output_report()
    
    print("=== Top 15 Molecules (Ranked) ===")
    print(output[[
        'rank', 'molecule', 'ensemble_score', 'investment_tier',
        'market_size_$M', 'revenue_growth_%', 'confidence_level'
    ]].head(15).to_string(index=False))
    
    print("\n=== Investment Tier Summary ===")
    print(output['investment_tier'].value_counts())
    
    print("\n=== Portfolio Recommendation ===")
    portfolio = formatter.recommend_portfolio_balance(output)
    print(portfolio['recommendation'])
    
    print("\n=== Summary Stats ===")
    summary = formatter.export_summary_json()
    print(f"Total molecules: {summary['total_molecules']}")
    print(f"High potential: {summary['investment_tiers']['high_potential']['count']}")
    print(f"Moderate: {summary['investment_tiers']['moderate']['count']}")
    print(f"Avoid: {summary['investment_tiers']['avoid']['count']}")
