import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
import sys

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import predict_batch, check_api_health, generate_forecast_dates
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Sales Forecast Inference",
    page_icon="🔮",
    layout="wide"
)

# Initialize session state
if 'api_connected' not in st.session_state:
    st.session_state.api_connected = False

# Header
st.title("🔮 Sales Forecast Inference")
st.markdown("Generate sales predictions using trained ML models")

# Sidebar for API status
with st.sidebar:
    st.header("🔌 API Connection")
    
    # Check API status
    if st.button("🔄 Check Connection", type="primary", use_container_width=True):
        with st.spinner("Checking API..."):
            st.session_state.api_connected = check_api_health()
            if st.session_state.api_connected:
                st.success("✅ API Connected")
            else:
                st.error("❌ API Offline")
    
    if st.session_state.api_connected:
        st.success("✅ API Connected")
    else:
        st.warning("⚠️ API not verified")
    
    st.markdown("---")
    
    # Forecast settings
    st.header("⚙️ Settings")
    
    st.markdown("**Quick Presets:**")
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    with preset_col1:
        if st.button("📅 1 Week", use_container_width=True):
            st.session_state.preset_days = 7
    with preset_col2:
        if st.button("📅 1 Month", use_container_width=True):
            st.session_state.preset_days = 30
    with preset_col3:
        if st.button("📅 3 Months", use_container_width=True):
            st.session_state.preset_days = 90

# Main content
if st.session_state.api_connected:
    st.subheader("📊 Generate Predictions")
    
    # Input configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        store_id = st.text_input(
            "Store ID",
            value="store_001",
            help="Enter the store ID for prediction"
        )
    
    with col2:
        prediction_date = st.date_input(
            "Start Date",
            value=datetime.now(),
            help="Date to start predictions from"
        )
    
    with col3:
        forecast_days = st.number_input(
            "Forecast Days",
            min_value=1,
            max_value=90,
            value=30,
            help="Number of days to forecast"
        )
    
    # Generate predictions
    st.markdown("---")
    
    # Center the button only
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_clicked = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)
    
    # Results displayed at full width
    if generate_clicked:
        with st.spinner("Generating forecast..."):
            try:
                # Generate forecast dates
                start_date = prediction_date.strftime("%Y-%m-%d")
                forecast_dates = generate_forecast_dates(start_date, forecast_days)
                
                # Prepare batch request
                requests_data = [
                    {"store_id": store_id, "date": date}
                    for date in forecast_dates
                ]
                
                # Call API
                results = predict_batch(requests_data)
                
                st.success("✅ Forecast generated successfully!")
                
                # Show metrics (use first result for metadata)
                st.markdown("### 📈 Forecast Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Store ID", results[0].get('store_id', store_id) if results else store_id)
                with col2:
                    st.metric("Forecast Period", f"{len(results)} days")
                with col3:
                    st.metric("Model Version", results[0].get('model_version', 'N/A') if results else 'N/A')
                
                # Display prediction details
                st.markdown("### 📊 Prediction Results")
                
                # Create visualization dataframe from list of results
                viz_data = []
                for result in results:
                    predictions = result.get('predictions', {})
                    intervals = result.get('intervals', {})
                    
                    # Get ensemble prediction (previous: ensemble_pred > xgboost_pred > lgb_pred)
                    pred_value = predictions.get('ensemble_pred', # try get ensemble
                                predictions.get('xgboost', # ensemble invalid, get xgboost
                                predictions.get('lgb_pred', 0))) # xgboost invalid, get lgb
                    
                    # Get ensemble intervals (or first available)
                    ensemble_intervals = intervals.get('ensemble_pred', 
                                        intervals.get('xgboost', 
                                        intervals.get('lgb_pred', {})))
                    
                    conf_95 = ensemble_intervals.get('confidence_0.95', [0, 0])
                    
                    viz_data.append({
                        'date': result.get('date'),
                        'prediction': pred_value,
                        'lower': conf_95[0] if isinstance(conf_95, list) and len(conf_95) > 0 else 0,
                        'upper': conf_95[1] if isinstance(conf_95, list) and len(conf_95) > 1 else 0
                    })
                
                viz_df = pd.DataFrame(viz_data)
                viz_df['date'] = pd.to_datetime(viz_df['date'])
                
                # Create plotly chart
                fig = go.Figure()
                
                # trace 1: Forecast line (prediction)
                fig.add_trace(go.Scatter(
                    x=viz_df['date'],
                    y=viz_df['prediction'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='green', width=3),
                    marker=dict(size=8)
                ))
                
                # trace 2: Confidence interval (upper prediction)
                fig.add_trace(go.Scatter(
                    x=viz_df['date'],
                    y=viz_df['upper'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,255,0,0)',
                    showlegend=False
                ))
                # trace 3: Confidence interval (lower prediction)
                fig.add_trace(go.Scatter(
                    x=viz_df['date'],
                    y=viz_df['lower'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,255,0,0.2)',
                    name='95% Confidence'
                ))
                
                fig.update_layout(
                    title=f"Sales Forecast - {store_id}",
                    xaxis_title="Date",
                    yaxis_title="Predicted Sales",
                    hovermode='x unified',
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show data table
                with st.expander("📋 View Detailed Predictions"):
                    st.dataframe(
                        viz_df.style.format({
                            'prediction': '{:.2f}',
                            'lower': '{:.2f}',
                            'upper': '{:.2f}'
                        }),
                        use_container_width=True
                    )
                
                # Download section
                st.markdown("### 💾 Export Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    csv = viz_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast (CSV)",
                        data=csv,
                        file_name=f"sales_forecast_{store_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col2:
                    st.info(f"🕐 {results[0].get('prediction_timestamp', 'N/A') if results else 'N/A'}")
            
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                logger.error(f"Prediction error: {str(e)}")
                
                # Show more details in expander
                with st.expander("🔍 Error Details"):
                    st.code(str(e))

else:
    # API not connected
    st.warning("⚠️ Please connect to the API server before making predictions.")
    st.info("👈 Click 'Check Connection' in the sidebar to verify the API is running")
    
    # Add helpful information
    with st.expander("ℹ️ API Server not running? Here's what to do:", expanded=True):
        st.markdown("""
        ### Start the API Server
        
        The prediction API needs to be running to make forecasts:
        
        1. **Navigate to the model serving directory**:
           ```bash
           cd ../Sales-Forecasting-Mlops/include/model_serving
           ```
        
        2. **Start the API server**:
           ```bash
           uvicorn controller:app --host 0.0.0.0 --port 8000 --reload
           ```
        
        3. **Verify the server is running**:
           - API should be available at: [http://localhost:8000](http://localhost:8000)
           - Check health endpoint: [http://localhost:8000/health](http://localhost:8000/health)
        
        4. **Come back here**:
           - Click "Check Connection" in the sidebar
           - Connection should succeed
        
        ### Environment Variables
        
        Make sure the `BACKEND_URL` environment variable is set correctly:
        - Default: `http://localhost:8000`
        - Set in `.env` file or environment
        
        ### Troubleshooting
        
        - Ensure the inference service has models loaded
        - Check that port 8000 is not in use by another service
        - Verify firewall settings allow local connections
        """)