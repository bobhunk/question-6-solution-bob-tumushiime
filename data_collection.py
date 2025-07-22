import requests
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

class AfricanEducationDataCollector:
    """
    A tool to gather education data from African countries using the World Bank API.
    
    This class helps us understand education challenges by collecting information about
    school enrollment, completion rates, gender gaps, and economic factors that affect
    whether children stay in school or drop out.
    """
    
    def __init__(self):
        self.base_url_wb = "https://api.worldbank.org/v2"
        self.african_countries = {
            # Sub-Saharan Africa focus countries
            'UGA': 'Uganda',
            'KEN': 'Kenya', 
            'TZA': 'Tanzania',
            'RWA': 'Rwanda',
            'ETH': 'Ethiopia',
            'GHA': 'Ghana',
            'NGA': 'Nigeria',
            'ZAF': 'South Africa',
            'BWA': 'Botswana',
            'ZWE': 'Zimbabwe',
            'ZMB': 'Zambia',
            'MWI': 'Malawi',
            'MOZ': 'Mozambique',
            'MDG': 'Madagascar',
            'SEN': 'Senegal',
            'MLI': 'Mali',
            'BFA': 'Burkina Faso',
            'NER': 'Niger',
            'TCD': 'Chad',
            'CMR': 'Cameroon',
            'CAF': 'Central African Republic',
            'COD': 'Democratic Republic of Congo',
            'AGO': 'Angola',
            'NAM': 'Namibia',
            'LSO': 'Lesotho',
            'SWZ': 'Eswatini',
            'BDI': 'Burundi'
        }
        
        # Key education indicators for dropout analysis
        self.education_indicators = {
            # Enrollment rates
            'SE.PRM.NENR': 'Primary school enrollment, net (%)',
            'SE.SEC.NENR': 'Secondary school enrollment, net (%)',
            'SE.PRM.NENR.FE': 'Primary school enrollment, net (% female)',
            'SE.SEC.NENR.FE': 'Secondary school enrollment, net (% female)',
            'SE.PRM.NENR.MA': 'Primary school enrollment, net (% male)',
            'SE.SEC.NENR.MA': 'Secondary school enrollment, net (% male)',
            
            # Completion rates
            'SE.PRM.CMPT.ZS': 'Primary completion rate, total (% of relevant age group)',
            'SE.PRM.CMPT.FE.ZS': 'Primary completion rate, female (% of relevant age group)',
            'SE.PRM.CMPT.MA.ZS': 'Primary completion rate, male (% of relevant age group)',
            'SE.SEC.CMPT.LO.ZS': 'Lower secondary completion rate, total (%)',
            'SE.SEC.CMPT.LO.FE.ZS': 'Lower secondary completion rate, female (%)',
            'SE.SEC.CMPT.LO.MA.ZS': 'Lower secondary completion rate, male (%)',
            
            # Dropout and survival rates
            'SE.PRM.DURS': 'Primary school duration (years)',
            'SE.SEC.DURS': 'Secondary school duration (years)',
            'SE.PRM.PRSL.ZS': 'Persistence to last grade of primary, total (%)',
            'SE.PRM.PRSL.FE.ZS': 'Persistence to last grade of primary, female (%)',
            'SE.PRM.PRSL.MA.ZS': 'Persistence to last grade of primary, male (%)',
            
            # Out-of-school rates
            'SE.PRM.OOSC': 'Out-of-school children, primary school age',
            'SE.PRM.OOSC.FE': 'Out-of-school children, primary school age, female',
            'SE.PRM.OOSC.MA': 'Out-of-school children, primary school age, male',
            'SE.SEC.OOSC': 'Out-of-school adolescents, lower secondary school age',
            'SE.SEC.OOSC.FE': 'Out-of-school adolescents, lower secondary school age, female',
            'SE.SEC.OOSC.MA': 'Out-of-school adolescents, lower secondary school age, male',
            
            # Economic and social indicators
            'SE.XPD.TOTL.GD.ZS': 'Government expenditure on education, total (% of GDP)',
            'SE.XPD.PRIM.PC.ZS': 'Government expenditure per student, primary (% of GDP per capita)',
            'SE.XPD.SECO.PC.ZS': 'Government expenditure per student, secondary (% of GDP per capita)',
            'NY.GDP.PCAP.CD': 'GDP per capita (current US$)',
            'SI.POV.NAHC': 'Poverty headcount ratio at national poverty lines (% of population)',
            'SP.RUR.TOTL.ZS': 'Rural population (% of total population)',
            'SP.POP.TOTL': 'Population, total',
            'SP.DYN.TFRT.IN': 'Fertility rate, total (births per woman)',
            
            # Gender and literacy
            'SE.ADT.LITR.ZS': 'Literacy rate, adult total (% of people ages 15 and above)',
            'SE.ADT.LITR.FE.ZS': 'Literacy rate, adult female (% of females ages 15 and above)',
            'SE.ADT.LITR.MA.ZS': 'Literacy rate, adult male (% of males ages 15 and above)',
        }
        
        self.data_years = list(range(2010, 2025))  # 15-year analysis period (2010-2024)
        
    def fetch_worldbank_indicator(self, indicator_code, countries_list=None):
        """
        Fetch specific education indicator from World Bank API
        
        Args:
            indicator_code (str): World Bank indicator code
            countries_list (list): List of country codes (default: all African countries)
            
        Returns:
            pandas.DataFrame: Processed indicator data
        """
        if countries_list is None:
            countries_list = list(self.african_countries.keys())
            
        countries_str = ';'.join(countries_list)
        
        url = f"{self.base_url_wb}/country/{countries_str}/indicator/{indicator_code}"
        params = {
            'format': 'json',
            'date': f"{min(self.data_years)}:{max(self.data_years)}",
            'per_page': 5000
        }
        
        try:
            print(f"Fetching {indicator_code}: {self.education_indicators.get(indicator_code, 'Unknown indicator')}")
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if len(data) > 1 and data[1]:
                    records = []
                    for record in data[1]:
                        if record.get('value') is not None:
                            records.append({
                                'country_code': record.get('countryiso3code'),
                                'country_name': record.get('country', {}).get('value'),
                                'indicator_code': indicator_code,
                                'indicator_name': self.education_indicators.get(indicator_code, 'Unknown'),
                                'year': int(record.get('date')),
                                'value': float(record.get('value')),
                                'data_source': 'World Bank'
                            })
                    
                    if records:
                        df = pd.DataFrame(records)
                        print(f"  Retrieved {len(df)} records for {len(df['country_code'].unique())} countries")
                        return df
                    else:
                        print(f"  No data available for {indicator_code}")
                        return pd.DataFrame()
                else:
                    print(f"  No data returned for {indicator_code}")
                    return pd.DataFrame()
            else:
                print(f"  API request failed with status {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"  Error fetching {indicator_code}: {str(e)}")
            return pd.DataFrame()
    
    def collect_all_education_data(self):
        """
        Collect all education indicators for African countries
        
        Returns:
            pandas.DataFrame: Comprehensive education dataset
        """
        print("Starting comprehensive African education data collection...")
        print(f"Target countries: {len(self.african_countries)}")
        print(f"Education indicators: {len(self.education_indicators)}")
        print(f"Analysis period: {min(self.data_years)}-{max(self.data_years)}")
        print("-" * 60)
        
        all_data = []
        successful_indicators = 0
        
        for i, (indicator_code, indicator_name) in enumerate(self.education_indicators.items(), 1):
            print(f"[{i}/{len(self.education_indicators)}] Processing indicator...")
            
            df = self.fetch_worldbank_indicator(indicator_code)
            
            if not df.empty:
                all_data.append(df)
                successful_indicators += 1
            
            # Rate limiting - be respectful to the API
            time.sleep(0.5)
        
        if all_data:
            print("-" * 60)
            print("Combining all indicator data...")
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Data quality summary
            print(f"Data collection completed successfully!")
            print(f"Total records: {len(combined_df):,}")
            print(f"Successful indicators: {successful_indicators}/{len(self.education_indicators)}")
            print(f"Countries with data: {len(combined_df['country_code'].unique())}")
            print(f"Years covered: {combined_df['year'].min()}-{combined_df['year'].max()}")
            
            return combined_df
        else:
            print("No data collected. Please check API connectivity.")
            return pd.DataFrame()
    
    def create_dropout_risk_features(self, df):
        """
        Engineer features specifically for dropout risk prediction
        
        Args:
            df (pandas.DataFrame): Raw education data
            
        Returns:
            pandas.DataFrame: Feature-engineered dataset for ML modeling
        """
        print("Engineering dropout risk features...")
        
        # Pivot data for feature engineering
        pivot_df = df.pivot_table(
            index=['country_code', 'country_name', 'year'],
            columns='indicator_code',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        # Calculate derived features
        feature_df = pivot_df.copy()
        
        # Gender parity indices
        if 'SE.PRM.NENR.FE' in feature_df.columns and 'SE.PRM.NENR.MA' in feature_df.columns:
            feature_df['primary_gender_parity'] = feature_df['SE.PRM.NENR.FE'] / feature_df['SE.PRM.NENR.MA']
        
        if 'SE.SEC.NENR.FE' in feature_df.columns and 'SE.SEC.NENR.MA' in feature_df.columns:
            feature_df['secondary_gender_parity'] = feature_df['SE.SEC.NENR.FE'] / feature_df['SE.SEC.NENR.MA']
        
        # Transition rates (proxy for dropout risk)
        if 'SE.PRM.NENR' in feature_df.columns and 'SE.SEC.NENR' in feature_df.columns:
            feature_df['primary_to_secondary_transition'] = feature_df['SE.SEC.NENR'] / feature_df['SE.PRM.NENR']
        
        # Completion gaps
        if 'SE.PRM.NENR' in feature_df.columns and 'SE.PRM.CMPT.ZS' in feature_df.columns:
            feature_df['primary_completion_gap'] = feature_df['SE.PRM.NENR'] - feature_df['SE.PRM.CMPT.ZS']
        
        # Economic stress indicators
        if 'SE.XPD.TOTL.GD.ZS' in feature_df.columns and 'NY.GDP.PCAP.CD' in feature_df.columns:
            feature_df['education_investment_per_capita'] = (
                feature_df['SE.XPD.TOTL.GD.ZS'] * feature_df['NY.GDP.PCAP.CD'] / 100
            )
        
        # Risk categorization (target variable for ML)
        # Based on completion rates and out-of-school children
        risk_conditions = []
        
        if 'SE.PRM.CMPT.ZS' in feature_df.columns:
            risk_conditions.append(feature_df['SE.PRM.CMPT.ZS'] < 70)  # Low primary completion
        
        if 'SE.SEC.CMPT.LO.ZS' in feature_df.columns:
            risk_conditions.append(feature_df['SE.SEC.CMPT.LO.ZS'] < 50)  # Low secondary completion
        
        if 'SE.PRM.OOSC' in feature_df.columns:
            risk_conditions.append(feature_df['SE.PRM.OOSC'] > 100000)  # High out-of-school numbers
        
        if risk_conditions:
            # High risk if any condition is met
            feature_df['dropout_risk_high'] = np.where(
                pd.concat(risk_conditions, axis=1).any(axis=1), 1, 0
            )
        else:
            feature_df['dropout_risk_high'] = 0
        
        print(f"Feature engineering completed. Dataset shape: {feature_df.shape}")
        
        return feature_df
    
    def save_datasets(self, raw_df, features_df):
        """
        Save collected datasets to CSV files
        
        Args:
            raw_df (pandas.DataFrame): Raw education data
            features_df (pandas.DataFrame): Feature-engineered data
        """
        # Create data directory
        os.makedirs('data', exist_ok=True)
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        # Save raw data
        raw_filename = f"data/raw/african_education_raw_{datetime.now().strftime('%Y%m%d')}.csv"
        raw_df.to_csv(raw_filename, index=False)
        print(f"Raw data saved: {raw_filename}")
        
        # Save processed features
        features_filename = f"data/processed/dropout_risk_features_{datetime.now().strftime('%Y%m%d')}.csv"
        features_df.to_csv(features_filename, index=False)
        print(f"Feature data saved: {features_filename}")
        
        # Save metadata
        metadata = {
            'collection_date': datetime.now().isoformat(),
            'countries_included': list(self.african_countries.keys()),
            'indicators_collected': list(self.education_indicators.keys()),
            'years_covered': self.data_years,
            'raw_records': len(raw_df),
            'feature_records': len(features_df),
            'data_source': 'World Bank Education Statistics API'
        }
        
        metadata_filename = f"data/metadata_{datetime.now().strftime('%Y%m%d')}.json"
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved: {metadata_filename}")
        
        return raw_filename, features_filename, metadata_filename

def main():
    """
    Main execution function for data collection
    """
    print("=" * 80)
    print("AFRICAN EDUCATION DROPOUT RISK DATA COLLECTION")
    print("=" * 80)
    print()
    
    # Initialize collector
    collector = AfricanEducationDataCollector()
    
    # Collect raw education data
    raw_data = collector.collect_all_education_data()
    
    if not raw_data.empty:
        # Engineer features for dropout prediction
        feature_data = collector.create_dropout_risk_features(raw_data)
        
        # Save datasets
        raw_file, features_file, metadata_file = collector.save_datasets(raw_data, feature_data)
        
        print()
        print("=" * 80)
        print("DATA COLLECTION SUMMARY")
        print("=" * 80)
        print(f"Raw education data: {len(raw_data):,} records")
        print(f"Feature dataset: {len(feature_data):,} records")
        print(f"Countries covered: {len(raw_data['country_code'].unique())}")
        print(f"Indicators collected: {len(raw_data['indicator_code'].unique())}")
        print(f"Time period: {raw_data['year'].min()}-{raw_data['year'].max()}")
        print()
        print("Files created:")
        print(f"  - {raw_file}")
        print(f"  - {features_file}")
        print(f"  - {metadata_file}")
        print()
        print("Next steps:")
        print("  1. Run EDA analysis: jupyter notebook eda_analysis.ipynb")
        print("  2. Train ML models: python dropout_predictor.py")
        print("  3. Create interactive story: python interactive_story.py")
        print("=" * 80)
        
    else:
        print("Data collection failed. Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
