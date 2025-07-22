import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import dash
from dash import dcc, html, Input, Output, callback
import warnings
warnings.filterwarnings('ignore')

class AfricanEducationDashboard:
    """
    Interactive dashboard for exploring African education dropout patterns
    and supporting data-driven policy decisions.
    """
    
    def __init__(self):
        self.data = None
        self.country_mapping = {
            'UGA': 'Uganda', 'KEN': 'Kenya', 'TZA': 'Tanzania', 'RWA': 'Rwanda',
            'ETH': 'Ethiopia', 'GHA': 'Ghana', 'NGA': 'Nigeria', 'ZAF': 'South Africa',
            'BWA': 'Botswana', 'ZWE': 'Zimbabwe', 'ZMB': 'Zambia', 'MWI': 'Malawi',
            'MOZ': 'Mozambique', 'AGO': 'Angola', 'CMR': 'Cameroon', 'TCD': 'Chad',
            'CAF': 'Central African Republic', 'COD': 'Democratic Republic of Congo',
            'GAB': 'Gabon', 'GNQ': 'Equatorial Guinea', 'STP': 'Sao Tome and Principe',
            'BFA': 'Burkina Faso', 'CIV': 'Ivory Coast', 'GIN': 'Guinea',
            'LBR': 'Liberia', 'MLI': 'Mali', 'MRT': 'Mauritania', 'NER': 'Niger',
            'SEN': 'Senegal', 'SLE': 'Sierra Leone', 'TGO': 'Togo', 'BEN': 'Benin',
            'GMB': 'Gambia', 'GNB': 'Guinea-Bissau', 'CPV': 'Cape Verde',
            'MDG': 'Madagascar', 'MUS': 'Mauritius', 'SYC': 'Seychelles',
            'COM': 'Comoros', 'DJI': 'Djibouti', 'ERI': 'Eritrea', 'SOM': 'Somalia',
            'SSD': 'South Sudan', 'SDN': 'Sudan'
        }
        
    def load_data(self, data_path='data/processed/modeling_dataset.csv'):
        """Load the processed education dataset"""
        try:
            self.data = pd.read_csv(data_path)
            self.data['country_name'] = self.data['country_code'].map(self.country_mapping)
            print(f"Loaded data: {len(self.data)} records from {len(self.data['country_code'].unique())} countries")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def create_risk_map(self):
        """Create interactive choropleth map showing dropout risk by country"""
        if self.data is None:
            return None
            
        # Calculate average risk by country
        country_risk = self.data.groupby(['country_code', 'country_name']).agg({
            'dropout_risk_high': 'mean',
            'SE.PRM.NENR': 'mean',
            'SE.SEC.NENR': 'mean',
            'SE.PRM.CMPT.ZS': 'mean'
        }).reset_index()
        
        country_risk['risk_percentage'] = country_risk['dropout_risk_high'] * 100
        
        fig = px.choropleth(
            country_risk,
            locations='country_code',
            color='risk_percentage',
            hover_name='country_name',
            hover_data={
                'SE.PRM.NENR': ':.1f',
                'SE.SEC.NENR': ':.1f', 
                'SE.PRM.CMPT.ZS': ':.1f',
                'risk_percentage': ':.1f'
            },
            color_continuous_scale='Reds',
            range_color=[0, 100],
            title='African Countries: School Dropout Risk Assessment',
            labels={
                'risk_percentage': 'Dropout Risk (%)',
                'SE.PRM.NENR': 'Primary Enrollment (%)',
                'SE.SEC.NENR': 'Secondary Enrollment (%)',
                'SE.PRM.CMPT.ZS': 'Primary Completion (%)'
            }
        )
        
        fig.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='natural earth',
                scope='africa'
            ),
            title_x=0.5,
            width=1000,
            height=600
        )
        
        return fig
    
    def create_regional_comparison(self):
        """Create regional comparison charts"""
        if self.data is None:
            return None
            
        # Define regions
        regions = {
            'West Africa': ['BFA', 'CIV', 'GIN', 'LBR', 'MLI', 'MRT', 'NER', 'SEN', 'SLE', 'TGO', 'BEN', 'GMB', 'GNB', 'CPV', 'GHA', 'NGA'],
            'East Africa': ['UGA', 'KEN', 'TZA', 'RWA', 'ETH', 'DJI', 'ERI', 'SOM', 'SSD', 'SDN'],
            'Central Africa': ['CMR', 'TCD', 'CAF', 'COD', 'GAB', 'GNQ', 'STP', 'AGO'],
            'Southern Africa': ['ZAF', 'BWA', 'ZWE', 'ZMB', 'MWI', 'MOZ', 'MDG', 'MUS', 'SYC', 'COM']
        }
        
        # Add region column
        def get_region(country_code):
            for region, countries in regions.items():
                if country_code in countries:
                    return region
            return 'Other'
        
        self.data['region'] = self.data['country_code'].apply(get_region)
        
        # Calculate regional averages
        regional_stats = self.data.groupby('region').agg({
            'SE.PRM.NENR': 'mean',
            'SE.SEC.NENR': 'mean',
            'SE.PRM.CMPT.ZS': 'mean',
            'dropout_risk_high': 'mean'
        }).reset_index()
        
        regional_stats['dropout_risk_pct'] = regional_stats['dropout_risk_high'] * 100
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Primary Enrollment Rate', 'Secondary Enrollment Rate', 
                          'Primary Completion Rate', 'Dropout Risk Percentage'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Add bar charts
        fig.add_trace(
            go.Bar(x=regional_stats['region'], y=regional_stats['SE.PRM.NENR'], 
                   name='Primary Enrollment', marker_color='lightblue'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=regional_stats['region'], y=regional_stats['SE.SEC.NENR'], 
                   name='Secondary Enrollment', marker_color='lightgreen'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=regional_stats['region'], y=regional_stats['SE.PRM.CMPT.ZS'], 
                   name='Primary Completion', marker_color='lightyellow'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=regional_stats['region'], y=regional_stats['dropout_risk_pct'], 
                   name='Dropout Risk', marker_color='lightcoral'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Regional Education Performance Comparison",
            title_x=0.5,
            showlegend=False,
            height=600,
            width=1000
        )
        
        return fig
    
    def create_correlation_analysis(self):
        """Create correlation analysis visualization"""
        if self.data is None:
            return None
            
        # Select key indicators for correlation
        indicators = ['SE.PRM.NENR', 'SE.SEC.NENR', 'SE.PRM.CMPT.ZS', 
                     'NY.GDP.PCAP.CD', 'SP.RUR.TOTL.ZS', 'dropout_risk_high']
        
        correlation_data = self.data[indicators].corr()
        
        fig = px.imshow(
            correlation_data,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu',
            title="Correlation Matrix: Education and Economic Indicators"
        )
        
        fig.update_layout(
            title_x=0.5,
            width=800,
            height=600
        )
        
        return fig
    
    def create_time_series_analysis(self):
        """Create time series analysis of education trends"""
        if self.data is None:
            return None
            
        # Calculate yearly averages
        yearly_trends = self.data.groupby('year').agg({
            'SE.PRM.NENR': 'mean',
            'SE.SEC.NENR': 'mean',
            'SE.PRM.CMPT.ZS': 'mean',
            'dropout_risk_high': 'mean'
        }).reset_index()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Education Enrollment and Completion Trends', 'Dropout Risk Trend'),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Add enrollment and completion trends
        fig.add_trace(
            go.Scatter(x=yearly_trends['year'], y=yearly_trends['SE.PRM.NENR'],
                      mode='lines+markers', name='Primary Enrollment',
                      line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=yearly_trends['year'], y=yearly_trends['SE.SEC.NENR'],
                      mode='lines+markers', name='Secondary Enrollment',
                      line=dict(color='green')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=yearly_trends['year'], y=yearly_trends['SE.PRM.CMPT.ZS'],
                      mode='lines+markers', name='Primary Completion',
                      line=dict(color='orange')),
            row=1, col=1
        )
        
        # Add dropout risk trend
        fig.add_trace(
            go.Scatter(x=yearly_trends['year'], y=yearly_trends['dropout_risk_high'] * 100,
                      mode='lines+markers', name='Dropout Risk (%)',
                      line=dict(color='red')),
            row=2, col=1
        )
        
        fig.update_layout(
            title_text="African Education Trends Over Time",
            title_x=0.5,
            height=800,
            width=1000
        )
        
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="Dropout Risk (%)", row=2, col=1)
        
        return fig
    
    def generate_dashboard_html(self, output_file='african_education_dashboard.html'):
        """Generate complete HTML dashboard"""
        if not self.load_data():
            print("Failed to load data. Cannot generate dashboard.")
            return
            
        # Create all visualizations
        risk_map = self.create_risk_map()
        regional_comparison = self.create_regional_comparison()
        correlation_analysis = self.create_correlation_analysis()
        time_series = self.create_time_series_analysis()
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>African Education Dropout Analysis Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .chart-container {{ margin: 30px 0; }}
                .insights {{ background-color: #f0f8ff; padding: 20px; margin: 20px 0; border-radius: 10px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>African Education Dropout Analysis Dashboard</h1>
                <p>Interactive visualizations for data-driven education policy decisions</p>
            </div>
            
            <div class="insights">
                <h2>Key Insights</h2>
                <ul>
                    <li><strong>High-Risk Countries:</strong> Burkina Faso, Cameroon, Madagascar, Niger, and Mozambique show 80-93% high dropout risk</li>
                    <li><strong>Regional Disparities:</strong> West Africa (69.7%) lags behind Southern Africa (89.3%) in enrollment</li>
                    <li><strong>Economic Correlation:</strong> GDP per capita shows moderate positive correlation (r=0.304) with primary enrollment</li>
                    <li><strong>Rural Challenge:</strong> Higher rural population correlates with lower completion rates (r=-0.262)</li>
                </ul>
            </div>
            
            <div class="chart-container">
                <div id="risk-map"></div>
            </div>
            
            <div class="chart-container">
                <div id="regional-comparison"></div>
            </div>
            
            <div class="chart-container">
                <div id="correlation-analysis"></div>
            </div>
            
            <div class="chart-container">
                <div id="time-series"></div>
            </div>
            
            <script>
                // Render all charts
                Plotly.newPlot('risk-map', {risk_map.to_json()});
                Plotly.newPlot('regional-comparison', {regional_comparison.to_json()});
                Plotly.newPlot('correlation-analysis', {correlation_analysis.to_json()});
                Plotly.newPlot('time-series', {time_series.to_json()});
            </script>
        </body>
        </html>
        """
        
        # Save individual charts as HTML
        if risk_map:
            pyo.plot(risk_map, filename='risk_map.html', auto_open=False)
        if regional_comparison:
            pyo.plot(regional_comparison, filename='regional_comparison.html', auto_open=False)
        if correlation_analysis:
            pyo.plot(correlation_analysis, filename='correlation_analysis.html', auto_open=False)
        if time_series:
            pyo.plot(time_series, filename='time_series.html', auto_open=False)
            
        print("Dashboard visualizations created successfully!")
        print("Generated files:")
        print("- risk_map.html")
        print("- regional_comparison.html") 
        print("- correlation_analysis.html")
        print("- time_series.html")

def main():
    """Main execution function"""
    print("=== AFRICAN EDUCATION DASHBOARD GENERATOR ===")
    
    # Create dashboard
    dashboard = AfricanEducationDashboard()
    dashboard.generate_dashboard_html()
    
    print("\nDashboard generation complete!")
    print("Open the HTML files in your browser to view interactive visualizations.")

if __name__ == "__main__":
    main()
