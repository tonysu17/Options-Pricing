# visuals/plotter.py
import plotly.graph_objects as go
import pandas as pd

def plot_vol_surface_3d(vol_data, title='Implied Volatility Surface'):
    """
    Plots an interactive 3D volatility surface using Plotly.
    
    Args:
        vol_data (pd.DataFrame): DataFrame with columns 'T', 'strike', and 'iv'.
        title (str): The title for the plot.
    """
    # Create a pivot table for the surface plot
    surface_pivot = vol_data.pivot_table(values='iv', index='strike', columns='T')
    
    fig = go.Figure(data=)
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Time to Maturity (Years)',
            yaxis_title='Strike Price',
            zaxis_title='Implied Volatility'
        ),
        autosize=False,
        width=800, height=800,
        margin=dict(l=65, r=50, b=65, t=90)
    )
    
    fig.show()