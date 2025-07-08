import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Solar Energy Tracker",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .season-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(254,107,139,0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(254,107,139,0.6);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .energy-tip {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'solar_data' not in st.session_state:
    st.session_state.solar_data = []
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'season_model' not in st.session_state:
    st.session_state.season_model = None

# Solar energy calculation functions
def calc_kwh_summer(irradiance, humidity, wind_speed, ambient_temp, tilt_angle):
    return (0.25 * irradiance - 0.05 * humidity + 0.02 * wind_speed + 0.1 * ambient_temp - 0.03 * abs(tilt_angle - 30))

def calc_kwh_winter(irradiance, humidity, wind_speed, ambient_temp, tilt_angle):
    return (0.18 * irradiance - 0.03 * humidity + 0.015 * wind_speed + 0.08 * ambient_temp - 0.02 * abs(tilt_angle - 30))

def calc_kwh_monsoon(irradiance, humidity, wind_speed, ambient_temp, tilt_angle):
    return (0.15 * irradiance - 0.1 * humidity + 0.01 * wind_speed + 0.05 * ambient_temp - 0.04 * abs(tilt_angle - 30))

def predict_season(irradiance, humidity, wind_speed, ambient_temp, tilt_angle):
    """Predict season based on weather conditions"""
    if irradiance > 600 and humidity < 50 and ambient_temp > 30:
        return 'summer'
    elif irradiance < 400 and humidity > 70:
        return 'monsoon'
    else:
        return 'winter'

def calculate_kwh_by_season(season, irradiance, humidity, wind_speed, ambient_temp, tilt_angle):
    """Calculate kWh based on detected season"""
    if season == 'summer':
        return calc_kwh_summer(irradiance, humidity, wind_speed, ambient_temp, tilt_angle)
    elif season == 'winter':
        return calc_kwh_winter(irradiance, humidity, wind_speed, ambient_temp, tilt_angle)
    else:  # monsoon
        return calc_kwh_monsoon(irradiance, humidity, wind_speed, ambient_temp, tilt_angle)

def generate_sample_data():
    """Generate sample solar data for demonstration"""
    np.random.seed(42)
    
    # Feature ranges for different seasons
    feature_ranges = {
        'summer': {
            'irradiance': (600, 1000),
            'humidity': (10, 50),
            'wind_speed': (0, 5),
            'ambient_temperature': (30, 45),
            'tilt_angle': (10, 40),
        },
        'winter': {
            'irradiance': (300, 700),
            'humidity': (30, 70),
            'wind_speed': (1, 6),
            'ambient_temperature': (5, 20),
            'tilt_angle': (10, 40),
        },
        'monsoon': {
            'irradiance': (100, 600),
            'humidity': (70, 100),
            'wind_speed': (2, 8),
            'ambient_temperature': (20, 35),
            'tilt_angle': (10, 40),
        }
    }
    
    data = []
    for season in ['summer', 'winter', 'monsoon']:
        for _ in range(50):  # 50 data points per season
            ranges = feature_ranges[season]
            irr = np.random.uniform(*ranges['irradiance'])
            hum = np.random.uniform(*ranges['humidity'])
            wind = np.random.uniform(*ranges['wind_speed'])
            temp = np.random.uniform(*ranges['ambient_temperature'])
            tilt = np.random.uniform(*ranges['tilt_angle'])
            
            kwh = calculate_kwh_by_season(season, irr, hum, wind, temp, tilt)
            
            data.append({
                'irradiance': round(irr, 2),
                'humidity': round(hum, 2),
                'wind_speed': round(wind, 2),
                'ambient_temperature': round(temp, 2),
                'tilt_angle': round(tilt, 2),
                'kwh': round(kwh, 2),
                'season': season,
                'date': datetime.now().date()
            })
    
    return pd.DataFrame(data)

# App Header
st.markdown('<div class="main-header">☀️ Smart Solar Energy Tracker</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🌤️ Weather Input")
st.sidebar.markdown("Enter current weather conditions:")

with st.sidebar.form("weather_form"):
    irradiance = st.slider("☀️ Solar Irradiance (W/m²)", 100, 1000, 500)
    humidity = st.slider("💧 Humidity (%)", 10, 100, 50)
    wind_speed = st.slider("🌪️ Wind Speed (m/s)", 0, 10, 3)
    ambient_temp = st.slider("🌡️ Temperature (°C)", 0, 50, 25)
    tilt_angle = st.slider("📐 Panel Tilt Angle (°)", 0, 90, 30)
    
    submitted = st.form_submit_button("🔮 Calculate Energy Output")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    if submitted:
        # Predict season and calculate energy
        predicted_season = predict_season(irradiance, humidity, wind_speed, ambient_temp, tilt_angle)
        predicted_kwh = calculate_kwh_by_season(predicted_season, irradiance, humidity, wind_speed, ambient_temp, tilt_angle)
        
        # Display results
        st.markdown(f"""
        <div class="prediction-card">
            <h2>🎯 Energy Prediction</h2>
            <h3>Season: {predicted_season.title()}</h3>
            <h1>{predicted_kwh:.2f} kWh</h1>
            <p>Estimated daily solar energy output</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add to session data
        new_entry = {
            'irradiance': irradiance,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'ambient_temperature': ambient_temp,
            'tilt_angle': tilt_angle,
            'kwh': round(predicted_kwh, 2),
            'season': predicted_season,
            'date': datetime.now().date()
        }
        st.session_state.solar_data.append(new_entry)
        
        # Energy efficiency tips
        if predicted_kwh > 150:
            tip = "🟢 Excellent conditions! Your solar panels are performing optimally."
        elif predicted_kwh > 100:
            tip = "🟡 Good conditions. Consider cleaning panels for better efficiency."
        else:
            tip = "🔴 Low output conditions. Check for shading or panel maintenance."
        
        st.markdown(f'<div class="energy-tip"><strong>💡 Tip:</strong> {tip}</div>', unsafe_allow_html=True)

with col2:
    st.subheader("📊 Performance Metrics")
    
    if st.session_state.solar_data:
        df = pd.DataFrame(st.session_state.solar_data)
        
        # Display metrics
        avg_kwh = df['kwh'].mean()
        max_kwh = df['kwh'].max()
        total_kwh = df['kwh'].sum()
        
        st.metric("Average Output", f"{avg_kwh:.1f} kWh")
        st.metric("Peak Output", f"{max_kwh:.1f} kWh")
        st.metric("Total Generated", f"{total_kwh:.1f} kWh")
        
        # Monthly savings calculation
        monthly_savings = total_kwh * 5  # Assuming ₹5 per kWh
        st.metric("💰 Estimated Savings", f"₹{monthly_savings:.0f}")

# Data visualization section
if st.session_state.solar_data:
    st.subheader("📈 Solar Performance Analytics")
    
    df = pd.DataFrame(st.session_state.solar_data)
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Daily Output", "🌍 Seasonal Analysis", "🔧 Factor Impact", "🤖 ML Predictions"])
    
    with tab1:
        # Daily output chart
        fig_daily = px.line(df, x='date', y='kwh', color='season',
                           title='Daily Solar Energy Output',
                           labels={'kwh': 'Energy Output (kWh)', 'date': 'Date'})
        fig_daily.update_layout(height=400)
        st.plotly_chart(fig_daily, use_container_width=True)
    
    with tab2:
        # Seasonal analysis
        seasonal_stats = df.groupby('season')['kwh'].agg(['mean', 'max', 'min']).reset_index()
        
        fig_season = px.box(df, x='season', y='kwh', color='season',
                           title='Energy Output Distribution by Season')
        st.plotly_chart(fig_season, use_container_width=True)
        
        st.dataframe(seasonal_stats, use_container_width=True)
    
    with tab3:
        # Factor impact analysis
        col1, col2 = st.columns(2)
        
        with col1:
            fig_irr = px.scatter(df, x='irradiance', y='kwh', color='season',
                               title='Energy vs Solar Irradiance')
            st.plotly_chart(fig_irr, use_container_width=True)
        
        with col2:
            fig_temp = px.scatter(df, x='ambient_temperature', y='kwh', color='season',
                                title='Energy vs Temperature')
            st.plotly_chart(fig_temp, use_container_width=True)
    
    with tab4:
        # Machine Learning predictions
        if len(df) > 10:
            X = df[['irradiance', 'humidity', 'wind_speed', 'ambient_temperature', 'tilt_angle']]
            y = df['kwh']
            
            # Train model
            if st.button("🚀 Train ML Model"):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                st.session_state.trained_model = model
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("R² Score", f"{r2:.3f}")
                with col2:
                    st.metric("MSE", f"{mse:.3f}")
                
                # Actual vs Predicted plot
                fig_ml = px.scatter(x=y_test, y=y_pred,
                                  title='Actual vs Predicted Energy Output',
                                  labels={'x': 'Actual kWh', 'y': 'Predicted kWh'})
                
                # Add diagonal line
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                fig_ml.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                               line=dict(color="red", dash="dash"))
                
                st.plotly_chart(fig_ml, use_container_width=True)
                
                # Feature importance
                importance = abs(model.coef_)
                features = ['Irradiance', 'Humidity', 'Wind Speed', 'Temperature', 'Tilt Angle']
                
                fig_importance = px.bar(x=features, y=importance,
                                      title='Feature Importance in Energy Prediction')
                st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.info("📊 Add more data points to enable ML predictions!")

# Data management section
st.subheader("📁 Data Management")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Load Sample Data"):
        sample_data = generate_sample_data()
        st.session_state.solar_data.extend(sample_data.to_dict('records'))
        st.success("✅ Sample data loaded successfully!")

with col2:
    if st.button("🗑️ Clear All Data"):
        st.session_state.solar_data = []
        st.success("✅ All data cleared!")

with col3:
    if st.session_state.solar_data:
        df = pd.DataFrame(st.session_state.solar_data)
        csv = df.to_csv(index=False)
        st.download_button(
            label="💾 Download Data",
            data=csv,
            file_name=f'solar_data_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv'
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin-top: 2rem;'>
    <h3>🌱 Sustainable Energy Tracking</h3>
    <p>Monitor your solar panel performance and optimize energy generation with real-time analytics and ML predictions.</p>
</div>
""", unsafe_allow_html=True)