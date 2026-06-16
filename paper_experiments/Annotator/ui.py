import plotly.express as px
from dash import Dash, dcc, html, Input, Output, no_update, callback, State
import pandas as pd
import plotly.graph_objects as go
import cv2
import os
import shutil

DATA_PATH = "filtered_dataset_x1y1x2y2.csv"  
IMAGE_DIR = "Pre_processing_dataset/Wider_merged"

##### INITITALIZATION 
if not os.path.exists(DATA_PATH):
    print("in")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    path = os.path.join(project_root, "preProcessing_dataset/filtered_dataset_x1y1x2y2.csv") 
    print(path)
    if os.path.exists(path):
        saving_path = os.path.join(script_dir, DATA_PATH )
        shutil.copy(path, saving_path)
    else:
        raise FileNotFoundError(f"{DATA_PATH} not found in current directory or fallback location.")
    df = pd.read_csv(saving_path)
    if "Sex" not in df.columns:
        df["Sex"] = pd.NA  

    if "Ethnicity" not in df.columns:
        df["Ethnicity"] = pd.NA  

    if "Valid" not in df.columns:
        df["Valid"] = True  
else : 
    df = pd.read_csv(DATA_PATH)

    if "Sex" not in df.columns:
        df["Sex"] = pd.NA  

    if "Ethnicity" not in df.columns:
        df["Ethnicity"] = pd.NA  

    if "Valid" not in df.columns:
        df["Valid"] = True  

##### HELPER METHOD
def encode_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def create_figure(image_path, x1, y1, x2, y2):
    # Read and encode the image
    image = encode_image(image_path)
    if image is None:
        return go.Figure()
    
    ### Calculate padding: 1/3 of the width and height of the bounding box
    padding_x = (x2 - x1) / 3
    padding_y = (y2 - y1) / 3
    
    ### Apply padding to the coordinates
    x1_padded = max(x1 - padding_x, 0)  
    y1_padded = max(y1 - padding_y, 0)  
    x2_padded = x2 + padding_x  
    y2_padded = y2 + padding_y  

    fig = px.imshow(image)
    fig.update_layout(dragmode="drawrect")

    fig.add_shape(
        type="rect",
        x0=x1_padded, y0=y1_padded, x1=x2_padded, y1=y2_padded,
        line=dict(color="red", width=1)
    )

    return fig

##### ANNOTATION RESTART FROM LAST NOT ANNOTATED ROW
def find_first_unannotated():
    for i, row in df.iterrows():
        if pd.isna(row["Sex"]) or pd.isna(row["Ethnicity"]):
            return i 
    return 0  

app = Dash()
app.layout = html.Div(
    [
        html.H3("Image annotation"),
        
        ### Progress Indicator
        html.Div(id="progress-indicator", style={'fontSize': '16px', 'fontWeight': 'bold'}),

        dcc.Graph(id="graph-picture", config={'scrollZoom': True}),

        html.Label("Characteristics"),
        dcc.Markdown("Select sex and ethnicity annotations"),

        html.Label("Sex"),
        dcc.Dropdown(id='sex-dropdown', options=[
            {'label': 'Male', 'value': 'Male'},
            {'label': 'Female', 'value': 'Female'},
            {'label': 'Undetermined', 'value': 'Undetermined'}
        ], placeholder='Select Sex'),

        html.Label("Ethnicity"),
        dcc.Dropdown(id='ethnicity-dropdown', options=[
            {'label': 'White', 'value': 'White'},
            {'label': 'Black', 'value': 'Black'},
            {'label': 'Asian', 'value': 'Asian'},
            {'label': 'Indian', 'value': 'Indian'},
            {'label': 'Middle Eastern', 'value': 'Middle Eastern'},
            {'label': 'Other', 'value': 'Other'},
            {'label': 'Undetermined', 'value': 'Undetermined'}

        ], placeholder='Select Ethnicity'),

        html.Label("Is this image valid?"),
        dcc.Dropdown(id='valid-dropdown', options=[
            {'label': 'Valid', 'value': True},
            {'label': 'Not Valid', 'value': False}
        ], placeholder='Select Validity'),

        html.Button("Save Annotation & Next", id='save-next-button', n_clicks=0),
        html.Button("Previous Image", id='previous-button', n_clicks=0),
        
        html.Div(id='ethnicity-images', style={'display': 'flex', 'flexWrap': 'nowrap', 'gap': '10px'}),
        dcc.Store(id='current-index', data=find_first_unannotated()),  
        html.Pre(id="annotations-data"),
    ]
)

@app.callback(
    Output('graph-picture', 'figure'),
    Output('current-index', 'data'),
    Output('save-next-button', 'n_clicks'),
    Output('previous-button', 'n_clicks'),
    Output('progress-indicator', 'children'), 
    Input('save-next-button', 'n_clicks'),
    Input('previous-button', 'n_clicks'),
    State('current-index', 'data')
)
def update_image(n_clicks_next, n_clicks_previous, index):
    total_rows = len(df)

    def find_next_unannotated_row(start_index):
        """Find the next unannotated row starting from the given index."""
        for i in range(start_index, len(df)):
            if pd.isna(df.loc[i, 'Sex']) or pd.isna(df.loc[i, 'Ethnicity']):  
                return i
        return -1 
    
    if index >= total_rows or index < 0:
        return no_update, no_update, 0, 0, no_update

    if n_clicks_next > 0: 
        index = find_next_unannotated_row(index+1 )
    elif n_clicks_previous > 0: 
        index = max(index - 1, 0)  

    row = df.iloc[index]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    image_path = os.path.join(project_root,IMAGE_DIR, row['filename'])

    progress_text = f" Progress: {index + 1} / {total_rows}"
    
    return create_figure(image_path, row['x1'], row['y1'], row['x2'], row['y2']), index, 0, 0, progress_text


@app.callback(
    Output("annotations-data", "children"),  
    Input('save-next-button', 'n_clicks'),
    Input("sex-dropdown", "value"),
    Input("ethnicity-dropdown", "value"),
    Input("valid-dropdown", "value"),
    State("current-index", "data"),
    prevent_initial_call=True,
)
def on_new_annotation(n_clicks, sex, ethnicity, valid, index):
    ### Save sex, ethnicity, and validity
    df.at[index, "Sex"] = sex
    df.at[index, "Ethnicity"] = ethnicity
    df.at[index, "Valid"] = valid
    df.to_csv(DATA_PATH, index=False)
    
    return f"Saved annotation: Sex={sex}, Ethnicity={ethnicity}, Valid={valid}"


@app.callback(
    Output('ethnicity-images', 'children'),
    Input('ethnicity-dropdown', 'value'),
    Input('sex-dropdown', 'value'),
    prevent_initial_call=True,
)
def display_filtered_images(ethnicity, sex):
    if not ethnicity or not sex:
        return []

    filtered_df = df[(df['Ethnicity'] == ethnicity) & (df['Sex'] == sex)]
    last_images = filtered_df.tail(5)  
    image_components = [
        html.Div(
            dcc.Graph(
                figure=create_figure(os.path.join(IMAGE_DIR, row['filename']), row['x1'], row['y1'], row['x2'], row['y2']),
                config={'displayModeBar': False}
            ),
            style={
                'flex': '1',  # Each image takes equal space
                'max-width': '20%',  # Each image takes up 20% of the row
                'height': '100vh',  # Full height of the viewport
                'margin': '0',  # Remove margin
                'padding': '0'  # Remove padding
            }
        )
        for _, row in last_images.iterrows()
    ]

    return html.Div(image_components, style={
        'display': 'flex',
        'flex-direction': 'row',  # Ensure images are in a single row
        'justify-content': 'space-between',  # Space out the images
        'align-items': 'center',
        'width': '100vw',  # Full width of the viewport
        'height': '100vh',  # Full height of the viewport
        'gap': '0px',  # No gap between images
        'margin': '0',  # Remove margin
        'padding': '0'  # Remove padding
    })

if __name__ == "__main__":
    app.run(debug=False)


