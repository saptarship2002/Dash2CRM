# --- Make sure to run this cell first: !pip install dash dash-bootstrap-components gunicorn pandas scikit-learn ---

from dash import Dash, dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# --- 1. Initialize the Dash App with a modern theme ---
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server
app.title = "Global Mineral Dashboard"

# --- 2. Load Data and Define Layout ---
try:
    # --- MODIFIED LINE: Load data from the provided GitHub URL ---
    data_url = 'https://raw.githubusercontent.com/saptarship2002/Mineral-Dashboard-ACPET/refs/heads/main/final%201.csv'
    df = pd.read_csv(data_url)
    # --- END OF MODIFICATION ---

    # --- Data Cleaning and Preparation ---\
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').dropna().astype(int)
    years = sorted(df['Year'].unique())
    home_country = "India"
    home_country_color = '#20c997'
    unit = "(tonnes)"

    # Get unique lists for dropdowns
    prod_minerals = sorted(df['Production Mineral'].dropna().unique())
    import_minerals = sorted(df['Import Mineral Name'].dropna().unique())
    all_minerals = sorted(list(set(prod_minerals + import_minerals)))
    all_minerals_with_total = ["--- All Minerals ---\"] + all_minerals

    # Identify numeric columns for indicators
    known_cols = ['Country', 'Year', 'Production Mineral', 'Production Qty', 'Import Mineral Name', 'Import Qty']
    indicator_cols = sorted([col for col in df.columns if col not in known_cols and pd.api.types.is_numeric_dtype(df[col])])

    # --- 3. Define App Layout Components ---

    # Header section
    header = html.Header(
        dbc.Container(
            dbc.Row(
                [
                    dbc.Col(html.Img(src=app.get_asset_url('logo.png'), height="50px"), width="auto"),
                    dbc.Col(
                        [
                            html.H1("Global Mineral Dashboard", className="m-0 fs-3"),
                            html.P("Analyze Production, Imports, and Strategic Indicators", className="m-0 text-muted small"),
                        ],
                        width=True,
                        className="d-flex flex-column justify-content-center",
                    ),
                    dbc.Col(
                        dbc.Button(
                            "About",
                            id="open-about-modal",
                            color="secondary",
                            outline=True,
                            className="float-end",
                        ),
                        width="auto",
                    ),
                ],
                align="center",
                className="py-3",
            ),
            fluid=True,
        ),
        className="bg-light border-bottom sticky-top",
    )

    # Control panel section
    controls = dbc.Card(
        [
            dbc.Row(
                [
                    dbc.Col([
                        dbc.Label("Select Year:", html_for="year-slider"),
                        dcc.Slider(
                            min=min(years),
                            max=max(years),
                            step=1,
                            value=max(years) - 1,
                            marks={str(y): str(y) for y in years if y % 2 == 0},
                            id="year-slider",
                        ),
                    ], width=12, md=6),
                    dbc.Col([
                        dbc.Label("Select Mineral:", html_for="mineral-dropdown"),
                        dcc.Dropdown(
                            id="mineral-dropdown",
                            options=[{"label": m, "value": m} for m in all_minerals_with_total],
                            value=all_minerals_with_total[0],
                        ),
                    ], width=12, md=6),
                ],
                className="g-3",
            )
        ],
        body=True,
        className="mb-4",
    )

    # Main dashboard tabs
    dashboard_tabs = dbc.Tabs(
        [
            dbc.Tab(label="Global Overview", tab_id="tab-overview"),
            dbc.Tab(label="Strategic Partner Analysis", tab_id="tab-analysis"),
        ],
        id="dashboard-tabs",
        active_tab="tab-overview",
        className="mb-4",
    )

    # Application layout
    app.layout = html.Div(
        [
            header,
            dbc.Container(
                [
                    controls,
                    dashboard_tabs,
                    dbc.Spinner(html.Div(id="tab-content"), color="primary"),
                    # --- About Modal ---
                    dbc.Modal(
                        [
                            dbc.ModalHeader(dbc.ModalTitle("About This Dashboard")),
                            dbc.ModalBody(
                                [
                                    html.P(
                                        "This dashboard provides insights into global mineral production and imports, with a special focus on strategic indicators for trade analysis."
                                    ),
                                    html.H5("Tabs:"),
                                    html.Ul(
                                        [
                                            html.Li(
                                                [html.Strong("Global Overview:"), " View worldwide mineral production and India's import data on interactive maps."]
                                            ),
                                            html.Li(
                                                [html.Strong("Strategic Partner Analysis:"), " Analyze potential trading partners for India based on production, import needs, and key economic/political indicators. A score is calculated to rank countries."]
                                            ),
                                        ]
                                    ),
                                    html.H5("How the Analysis Score is Calculated:"),
                                    html.P(
                                        "The score ranks countries as potential partners. A higher score is better. It is calculated using a weighted formula:"
                                    ),
                                    html.Pre(
                                        "Score = (w1 * Production) - (w2 * Import Need) + (w3 * WGI) + ...",
                                        className="bg-light p-2 rounded",
                                    ),
                                    html.Small(
                                        "Note: All indicators are normalized (scaled from 0 to 1) before scoring. Weights can be adjusted in the 'Strategic Partner Analysis' tab."
                                    ),
                                ]
                            ),
                            dbc.ModalFooter(
                                dbc.Button("Close", id="close-about-modal", className="ms-auto")
                            ),
                        ],
                        id="about-modal",
                        is_open=False,
                        size="lg",
                    ),
                ],
                fluid=True,
                className="py-4",
            ),
        ],
        className="bg-light-subtle",
    )

    # --- 4. Define Tab Content ---

    # --- Overview Tab ---
    overview_layout = dbc.Row(
        [
            dbc.Col(
                [
                    html.H4("Global Mineral Production", className="h5"),
                    dcc.Graph(id="production-map"),
                ],
                width=12,
                lg=6,
            ),
            dbc.Col(
                [
                    html.H4(f"{home_country}'s Mineral Imports", className="h5"),
                    dcc.Graph(id="import-map"),
                ],
                width=12,
                lg=6,
            ),
        ]
    )

    # --- Analysis Tab ---
    analysis_layout = dbc.Row(
        [
            # Analysis controls
            dbc.Col(
                dbc.Card(
                    [
                        html.H4("Analysis Configuration", className="h5"),
                        html.Hr(),
                        dbc.Label("Select Weights for Scoring:", className="fw-bold"),
                        html.P(
                            "Adjust the importance of each factor (0 = ignore, 5 = max importance).",
                            className="small text-muted",
                        ),
                        html.Div(id="indicator-sliders-container"),
                        dbc.Button(
                            "Run Analysis",
                            id="run-analysis-button",
                            color="primary",
                            className="w-100 mt-3",
                        ),
                    ],
                    body=True,
                ),
                width=12,
                lg=4,
                className="mb-4 mb-lg-0",
            ),
            # Analysis results
            dbc.Col(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Card(
                                    [
                                        html.H4("Analysis Map", className="h5"),
                                        dcc.Graph(id="analysis-map"),
                                    ],
                                    body=True,
                                ),
                                width=12,
                                className="mb-4",
                            ),
                            dbc.Col(
                                dbc.Card(
                                    [
                                        html.H4("Top 10 Potential Partners", className="h5"),
                                        dbc.Table(id="analysis-table", hover=True, striped=True, responsive=True),
                                    ],
                                    body=True,
                                ),
                                width=12,
                            ),
                        ]
                    )
                ],
                width=12,
                lg=8,
            ),
        ]
    )

except FileNotFoundError:
    app.layout = html.Div(
        [
            html.H1("Error: Data File Not Found", className="text-danger"),
            html.P("The application could not find the 'final 1.csv' file."),
            html.P("Please ensure the file is in the correct directory or the URL is accessible."),
        ]
    )
except Exception as e:
    app.layout = html.Div(
        [
            html.H1("An Error Occurred", className="text-danger"),
            html.P(f"An unexpected error occurred: {str(e)}"),
            html.P("Please check the data and the application code."),
        ]
    )

# --- 5. Callbacks ---

# --- Callback to switch tabs ---
@app.callback(
    Output("tab-content", "children"),
    Input("dashboard-tabs", "active_tab")
)
def render_tab_content(active_tab):
    if active_tab == "tab-overview":
        return overview_layout
    elif active_tab == "tab-analysis":
        return analysis_layout
    return html.P("This tab content is not available.")

# --- Callback for Overview Tab maps ---
@app.callback(
    [Output("production-map", "figure"), Output("import-map", "figure")],
    [Input("year-slider", "value"), Input("mineral-dropdown", "value")],
)
def update_overview_maps(selected_year, selected_mineral):
    # Filter data for the selected year
    df_year = df[df["Year"] == selected_year].copy()

    # Determine filter column based on mineral selection
    if selected_mineral == "--- All Minerals ---":
        prod_col = 'Production Qty'
        import_col = 'Import Qty'
        title_suffix = "All Minerals"
        df_prod = df_year.groupby('Country')[prod_col].sum().reset_index()
        df_import = df_year.groupby('Country')[import_col].sum().reset_index()
    else:
        prod_col = 'Production Qty'
        import_col = 'Import Qty'
        title_suffix = selected_mineral
        df_prod = df_year[df_year['Production Mineral'] == selected_mineral][['Country', prod_col]]
        df_import = df_year[df_year['Import Mineral Name'] == selected_mineral][['Country', import_col]]

    # --- Create Production Map ---
    prod_fig = go.Figure(go.Choropleth(
        locations=df_prod['Country'],
        z=df_prod[prod_col],
        locationmode='country names',
        colorscale='Greens',
        colorbar_title=unit,
        hovertemplate='<b>%{location}</b><br>Production: %{z:,.0f} ' + unit + '<extra></extra>'
    ))
    prod_fig.update_layout(
        title=f'Global {title_suffix} Production in {selected_year}',
        geo=dict(landcolor='#E5ECF6'),
        margin=dict(t=40, b=10, l=10, r=10)
    )

    # --- Create Import Map ---
    df_import_home = df_import[df_import['Country'] == home_country]
    
    # Handle case where home country has no imports for the selection
    if df_import_home.empty or df_import_home[import_col].sum() == 0:
        import_fig = go.Figure()
        import_fig.update_layout(
            title=f'{home_country} did not import {title_suffix} in {selected_year}',
            geo=dict(landcolor='#E5ECF6'),
            margin=dict(t=40, b=10, l=10, r=10)
        )
    else:
        # Highlight home country
        home_location = df_import_home.iloc[0]
        home_z = home_location[import_col]
        
        import_fig = go.Figure(go.Choropleth(
            locations=[home_location['Country']],
            z=[home_z],
            locationmode='country names',
            colorscale=[[0, home_country_color], [1, home_country_color]], # Single color
            showscale=False,
            hovertemplate='<b>%{location}</b><br>Imports: %{z:,.0f} ' + unit + '<extra></extra>'
        ))
        
        import_fig.update_layout(
            title=f'{home_country} {title_suffix} Imports in {selected_year} ({home_z:,.0f} {unit})',
            geo=dict(
                landcolor='#E5ECF6',
                showcountries=True,
                scope='world'
            ),
            margin=dict(t=40, b=10, l=10, r=10)
        )
        
    return prod_fig, import_fig

# --- Callback to dynamically create indicator sliders ---
@app.callback(
    Output("indicator-sliders-container", "children"),
    Input("dashboard-tabs", "active_tab") # Trigger when tab becomes active
)
def create_indicator_sliders(active_tab):
    if active_tab != 'tab-analysis':
        return no_update # Don't update if tab is not active

    default_weights = {
        'Production Qty': 5,
        'Import Qty': -5,
        'WGI Average': 3
    }
    
    sliders = []
    for col in ['Production Qty', 'Import Qty'] + indicator_cols:
        default_val = default_weights.get(col, 1) # Default to 1 if not in map
        sliders.append(
            html.Div([
                dbc.Label(col, html_for=f"slider-{col}"),
                dcc.Slider(
                    id={'type': 'indicator-slider', 'index': col},
                    min=-5,
                    max=5,
                    step=1,
                    value=default_val,
                    marks={i: str(i) for i in range(-5, 6)}
                )
            ], className="mb-3")
        )
    return sliders

# --- Callback for Analysis Tab ---
@app.callback(
    [Output("analysis-map", "figure"), Output("analysis-table", "children")],
    [Input("run-analysis-button", "n_clicks")],
    [State("year-slider", "value"),
     State("mineral-dropdown", "value"),
     State({'type': 'indicator-slider', 'index': dcc.ALL}, 'id'),
     State({'type': 'indicator-slider', 'index': dcc.ALL}, 'value')]
)
def update_analysis(n_clicks, selected_year, selected_mineral, slider_ids, slider_values):
    if n_clicks is None:
        # Default state before button is clicked
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Run analysis to see results", geo=dict(landcolor='#E5ECF6'), margin=dict(t=40, b=10, l=10, r=10))
        empty_table = [html.Thead(html.Tr([html.Th("Rank"), html.Th("Country"), html.Th("Score")]))]
        return empty_fig, empty_table

    if selected_mineral == "--- All Minerals ---":
        # Handle case where "All Minerals" is selected
        error_fig = go.Figure()
        error_fig.update_layout(title="Please select a specific mineral for analysis", geo=dict(landcolor='#E5ECF6'), margin=dict(t=40, b=10, l=10, r=10))
        error_table = [html.Thead(html.Tr([html.Th("Rank"), html.Th("Country"), html.Th("Score")])), html.Tbody([html.Tr([html.Td("N/A", colSpan=3, className="text-center")])])]
        return error_fig, error_table

    # --- 1. Prepare Data ---
    df_year = df[df["Year"] == selected_year].copy()
    
    # Get production data for the mineral
    df_prod = df_year[df_year['Production Mineral'] == selected_mineral][['Country', 'Production Qty']]
    if df_prod.empty:
        df_prod = df_year[['Country']].drop_duplicates()
        df_prod['Production Qty'] = 0
        
    # Get import data for the mineral
    df_import = df_year[df_year['Import Mineral Name'] == selected_mineral][['Country', 'Import Qty']]
    if df_import.empty:
        df_import = df_year[['Country']].drop_duplicates()
        df_import['Import Qty'] = 0

    # Get indicator data (one row per country for the selected year)
    indicator_data_cols = ['Country'] + indicator_cols
    df_indicators = df_year[indicator_data_cols].drop_duplicates(subset=['Country'])
    
    # Merge datasets
    analysis_df = pd.merge(df_indicators, df_prod, on='Country', how='left')
    analysis_df = pd.merge(analysis_df, df_import, on='Country', how='left')
    
    # Fill NaNs with 0 after merging (e.g., country has indicators but 0 production/import)
    analysis_df['Production Qty'] = analysis_df['Production Qty'].fillna(0)
    analysis_df['Import Qty'] = analysis_df['Import Qty'].fillna(0)
    
    # Remove the home country from the analysis
    analysis_df = analysis_df[analysis_df['Country'] != home_country].copy()
    
    # --- 2. Normalize Data ---
    scaler = MinMaxScaler()
    cols_to_scale = ['Production Qty', 'Import Qty'] + indicator_cols
    
    # Handle columns with 0 variance (e.g., all 0s)
    for col in cols_to_scale:
        if col in analysis_df.columns:
            if analysis_df[col].max() == analysis_df[col].min():
                # Set to 0 if all values are the same (prevents NaN from scaling)
                analysis_df[col] = 0.0
            else:
                # Scale non-NaN values
                valid_data = analysis_df[[col]].dropna()
                if not valid_data.empty:
                    scaled_data = scaler.fit_transform(valid_data)
                    analysis_df.loc[valid_data.index, col] = scaled_data.flatten()
    
    # Fill any remaining NaNs (e.g., in indicator columns) with 0.5 (neutral)
    analysis_df[cols_to_scale] = analysis_df[cols_to_scale].fillna(0.5)

    # --- 3. Calculate Score ---
    weights = {s_id['index']: s_val for s_id, s_val in zip(slider_ids, slider_values)}
    analysis_df['Score'] = 0
    
    for col, weight in weights.items():
        if col in analysis_df.columns:
            analysis_df['Score'] += analysis_df[col] * weight

    # --- 4. Create Outputs ---
    
    # Rank based on the score
    analysis_df['Rank'] = analysis_df['Score'].rank(ascending=False, method='dense').astype(int)
    analysis_df = analysis_df.sort_values('Rank')

    # Create the analysis map
    fig = go.Figure(go.Choropleth(
        locations=analysis_df['Country'],
        z=analysis_df['Rank'],
        locationmode='country names',
        colorscale='viridis_r', # Reversed viridis (low rank = better = purple)
        reversescale=True,
        colorbar_title='Rank',
        hovertemplate='<b>%{location}</b><br>Rank: %{z}<br>Score: %{customdata[0]:.3f}<extra></extra>',
        customdata=analysis_df[['Score']].values
    ))
    fig.update_layout(title=f'Top Trading Partners for {selected_mineral} in {selected_year}', geo=dict(landcolor='#E5ECF6'), margin=dict(t=40, b=10, l=10, r=10))

    # Create the results table for the Top 10
    top_10_df = analysis_df.head(10)
    table_header = [html.Thead(html.Tr([html.Th("Rank"), html.Th("Country"), html.Th("Score")]))]
    table_body = [html.Tbody([
        html.Tr([
            html.Td(row['Rank']),
            html.Td(row['Country']),
            html.Td(f"{row['Score']:.3f}")
        ]) for index, row in top_10_df.iterrows()
    ])]
    table = table_header + table_body
    
    return fig, table

# --- Callback for About Modal ---
@app.callback(
    Output("about-modal", "is_open"),
    [Input("open-about-modal", "n_clicks"), Input("close-about-modal", "n_clicks")],
    [State("about-modal", "is_open")],
)
def toggle_about_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# --- 6. Run the App ---
if __name__ == "__main__":
    app.run_server(debug=True)
