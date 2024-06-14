import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.interpolate import griddata
import plotly.graph_objs as go

# Import CSV data
@st.cache_data
def load_csv(file):
    return pd.read_csv(file, header=1, skiprows=[2, 3])

def calculate_outside(df):
    eau_per_min = 9.8615
    specific_heat = 4180
    total_area = 0.88 * 1.54
    
    df['Delta_T_InOut'] = df['T_out_H2O_Avg'] - df['T_in_H2O_Avg']
    df['Harvested_Joules'] = df['Delta_T_InOut'] * eau_per_min * specific_heat
    df['Harvested_m2'] = df['Harvested_Joules'] / (60 * total_area)
    df['DeltaINPUT_AIR'] = df['T_in_H2O_Avg'] - df['AirT_C_Avg']
    df['Efficiency'] = df['Harvested_m2']*100/df['SlrFD_W_Avg']
    return df

# Function to isolate data to specific dates
def filter_data_by_datetime(df, start_datetime, end_datetime):
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])  # Converts TIMESTAMP to datetime formating
    start_datetime = pd.to_datetime(start_datetime)  # Converts input of starting date and time to datetime formating
    end_datetime = pd.to_datetime(end_datetime)  # Converts input of ending date and time to datetime formating
    return df[(df['TIMESTAMP'] >= start_datetime) & (df['TIMESTAMP'] <= end_datetime)]  # Returns a DataFrame that only contains information from the specified period


def get_sensor_positions(sensor_part): # Function to get the spacial position of the sensors inside the prototype
    if sensor_part == "Upper Part":
        return {
            'Th_up_pos2_row1_Avg': (510, 180),
            'Th_up_pos4_row1_Avg': (1100, 180),
            'Th_up_pos6_row1_Avg': (1690, 180),
            'Th_up_pos2_row2_Avg': (410, 500),
            'Th_up_pos4_row2_Avg': (1000, 500),
            'Th_up_pos6_row2_Avg': (1590, 500),
            'Th_up_pos2_row3_Avg': (310, 820),
            'Th_up_pos4_row3_Avg': (900, 820),
            'Th_up_pos6_row3_Avg': (1490, 820)
        }
    else:
        return {
            'Th_down_pos2_row1_Avg': (510, 180),
            'Th_down_pos4_row1_Avg': (1100, 180),
            'Th_down_pos6_row1_Avg': (1690, 180),
            'Th_down_pos2_row2_Avg': (410, 500),
            'Th_down_pos4_row2_Avg': (1000, 500),
            'Th_down_pos6_row2_Avg': (1590, 500),
            'Th_down_pos2_row3_Avg': (310, 820),
            'Th_down_pos4_row3_Avg': (900, 820),
            'Th_down_pos6_row3_Avg': (1490, 820)
        }

# Generation of a animation showing the variance of temperature of the sensors in a day
def generate_daily_animation(filtered_df, sensor_positions):    
    if st.button("Animate"):
        # Grid to be ploted in the size of the rectangle that encompass all the sensors
        sensor_x = [pos[0] for pos in sensor_positions.values()]
        sensor_y = [pos[1] for pos in sensor_positions.values()]
        min_x, max_x = min(sensor_x), max(sensor_x)
        min_y, max_y = min(sensor_y), max(sensor_y)
        grid_x, grid_y = np.mgrid[min_x:max_x:200j, min_y:max_y:200j]
       
        frames = []

        timestamps = filtered_df['TIMESTAMP'].values
        for i in range(0, len(timestamps), 10):
            time = timestamps[i]
            time_df = filtered_df[filtered_df['TIMESTAMP'] == time]
            points = np.array(list(sensor_positions.values()))
            values = np.array([time_df[pos].mean() for pos in sensor_positions.keys()])
            grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
            heatmap = go.Heatmap(z=grid_z, x=np.linspace(min_x, max_x, 200), y=np.linspace(min_y, max_y, 200), colorscale='Viridis')
            frames.append(go.Frame(data=[heatmap], name=str(time)))
            
        sliders = [dict(
            steps=[dict(method='animate', args=[[frame.name], dict(mode='immediate', frame=dict(duration=300, redraw=True))], label=str(frame.name)) for frame in frames],
            transition=dict(duration=0),
            x=0.1,
            xanchor="left",
            y=0,
            yanchor="top"
        )]

        layout = go.Layout(
            sliders=sliders,
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=300, redraw=True), fromcurrent=True)])]
            )],
            xaxis=dict(title='X position (mm)', range=[min_x, max_x]),
            yaxis=dict(title='Y position (mm)', range=[min_y, max_y])
        )

        fig = go.Figure(data=frames[0].data if frames else [], layout=layout, frames=frames)
        st.plotly_chart(fig)


# Função principal
def main():
    st.title("Projet SunRoad")
    menu = st.sidebar.selectbox("Menu", ["Load Archives", "Heatmaps", "Plots", "Correlation"])

    if menu == "Load Archives":
        handle_file_upload()
    elif menu == "Heatmaps":
        visualize_heatmaps()
    elif menu == "Plots":
        visualize_plots()
    elif menu == "Correlation":
        visualize_correlation()

# Função para tratar o upload dos arquivos       
def handle_file_upload():
    st.header("Load Archives")
    uploaded_files = st.file_uploader("Upload the CSV documents", type="csv", accept_multiple_files=True)
    dataframes = []

    if uploaded_files:
        for file in uploaded_files:
            df = load_csv(file)
            print(df.info())
            #df = convert_columns_to_float(df)
            dataframes.append(df)

        if dataframes:
            # Realizando o merge conforme a lógica especificada
            merged_df = dataframes[0]
            for df in dataframes[1:]:
                merged_df = pd.merge(merged_df, df)   
            merged_df = calculate_outside(merged_df)
            st.write(merged_df)
            st.session_state['merged_df'] = merged_df
            print(merged_df.info()) 
# Função para visualizar os heatmaps
def visualize_heatmaps():
    st.header("Heatmaps")
    
    merged_df = st.session_state['merged_df']
    start_date = st.date_input("Select start date:", value=datetime.now().date(), key="start_heatmap_date")
    start_time = st.time_input("Select start time:", value=datetime.strptime("00:00", "%H:%M").time(), key="start_heatmap_time")
    end_date = st.date_input("Select end date:", value=datetime.now().date(), key="end_heatmap_date")
    end_time = st.time_input("Select end time:", value=datetime.strptime("23:59", "%H:%M").time(), key="end_heatmap_time")

    start_datetime = datetime.combine(start_date, start_time)
    end_datetime = datetime.combine(end_date, end_time)

    filtered_df = filter_data_by_datetime(merged_df, start_datetime, end_datetime)
    
    sensor_part = st.selectbox("Upper or Lower", ["Upper Part", "Lower Part"])
    sensor_positions = get_sensor_positions(sensor_part)
    visualization_mode = st.selectbox("Visualize", ["Interval average", "Animation of a day"])

    if visualization_mode == "Interval average":
        if st.button("Gerar Heatmap"):
            title = f"{sensor_part} Sensors Temperature Heatmap"
            interpolate_heatmap(filtered_df, sensor_positions, title)

    elif visualization_mode == "Animation of a day":
        generate_daily_animation(filtered_df, sensor_positions)

# Função para interpolar e criar o heatmap
def interpolate_heatmap(filtered_df, sensor_positions, title, std_threshold=3):
    # Calcular a média e o desvio padrão para cada sensor
    sensor_means = {pos: filtered_df[pos].mean() for pos in sensor_positions.keys()}
    sensor_stds = {pos: filtered_df[pos].std() for pos in sensor_positions.keys()}
    
    # Criar uma cópia do DataFrame filtrado para evitar alterações indesejadas
    filtered_df_copy = filtered_df.copy()
    
    # Aplicar o filtro para remover outliers
    for pos in sensor_positions.keys():
        mean = sensor_means[pos]
        std = sensor_stds[pos]
        lower_bound = mean - std_threshold * std
        upper_bound = mean + std_threshold * std
        filtered_df_copy[pos] = filtered_df_copy[pos].apply(lambda x: x if lower_bound <= x <= upper_bound else mean)
    
    # Restante do código para criar o heatmap permanece o mesmo
    sensor_x = [pos[0] for pos in sensor_positions.values()]
    sensor_y = [pos[1] for pos in sensor_positions.values()]
    min_x, max_x = min(sensor_x), max(sensor_x)
    min_y, max_y = min(sensor_y), max(sensor_y)
    grid_x, grid_y = np.mgrid[min_x:max_x:200j, min_y:max_y:200j]

    points = np.array(list(sensor_positions.values()))
    values = np.array([filtered_df_copy[pos].mean() for pos in sensor_positions.keys()])
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear')

    heatmap = go.Heatmap(z=grid_z, x=np.linspace(min_x, max_x, 200), y=np.linspace(min_y, max_y, 200), colorscale='Viridis')

    layout = go.Layout(title=title, xaxis=dict(title='X position (mm)'), yaxis=dict(title='Y position (mm)'))
    fig = go.Figure(data=[heatmap], layout=layout)
    st.plotly_chart(fig)

def visualize_plots():
    st.header("Graphical Plotting")
    
    merged_df = st.session_state['merged_df']
    columns = merged_df.columns.tolist()
    
    # Selecionar colunas para plotagem
    selected_columns = st.multiselect("Select the columns to plot", columns)
    
    # Verificar se foram selecionadas colunas suficientes para plotagem
    if len(selected_columns) < 1:
        st.warning("Please, select at least one column.")
        return
    
    start_date = st.date_input("Select start date:", value=datetime.now().date(), key="start_plot_date")
    start_time = st.time_input("Select start time:", value=datetime.strptime("00:00", "%H:%M").time(), key="start_plot_time")
    end_date = st.date_input("Select end date:", value=datetime.now().date(), key="end_plot_date")
    end_time = st.time_input("Select end time:", value=datetime.strptime("23:59", "%H:%M").time(), key="end_plot_time")
    title_plot = str(start_date)

    start_datetime = datetime.combine(start_date, start_time)
    end_datetime = datetime.combine(end_date, end_time)

    filtered_df = filter_data_by_datetime(merged_df, start_datetime, end_datetime)
    
    # Escolher se as colunas utilizarão eixo y principal ou secundário
    y_axes = {}
    for column in selected_columns:
        y_axes[column] = st.radio(f"Axis Y to {column}", ["Principal", "Secundary"])
    
    # Escolher simbologia para a legenda
    symbols = {}
    for column in selected_columns:
        symbols[column] = st.selectbox(f"Choose the symbol of {column}", ["Line", "Dotted", "Line and dash"])
    
    # Escolher se deseja plotar apenas valores positivos
    plot_positives_only = {}
    for column in selected_columns:
        plot_positives_only[column] = st.checkbox(f"Only positive values for {column}")
    
    # Botão para gerar os plots
    if st.button("Generate Plots"):
        fig = go.Figure()
        for column in selected_columns:
            if plot_positives_only[column]:
                positive_data = filtered_df[filtered_df[column] > 0]
                x_data = positive_data['TIMESTAMP']
                y_data = positive_data[column]
            else:
                x_data = filtered_df['TIMESTAMP']
                y_data = filtered_df[column]
            
            mode = 'lines+markers' if symbols[column] == "Line and dash" else 'lines' if symbols[column] == "Line" else 'markers'
            
            if y_axes[column] == "Principal":
                fig.add_trace(go.Scatter(x=x_data, y=y_data, mode=mode, name=column))
            else:
                fig.add_trace(go.Scatter(x=x_data, y=y_data, mode=mode, name=column, yaxis="y2"))
        
        # Layout do gráfico
        fig.update_layout(
            title=title_plot,
            xaxis_title="Time",
            yaxis_title="Value",
            yaxis=dict(title="Value", side="left"),
            yaxis2=dict(title="Value", side="right", overlaying="y"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Exibir o gráfico
        st.plotly_chart(fig)
        
        st.write(filtered_df[selected_columns].describe())
        eficiency_analysis(filtered_df)

def eficiency_analysis(merged_df):
    first_transition = None
    last_transition = None
    
    for i in range(1, len(merged_df)):
        actual_val = merged_df['Delta_T_InOut'].iloc[i]
        last_val = merged_df['Delta_T_InOut'].iloc[i - 1]
    
    # Verificando a transição de negativo para positivo
        if last_val < 0 and actual_val >= 0 and first_transition is None:
            first_transition = i
    
    # Verificando a transição de positivo para negativo
        if last_val >= 0 and actual_val < 0:
           last_transition = i
    
    start_time = pd.to_datetime(merged_df['TIMESTAMP'].iloc[first_transition])
    end_time = pd.to_datetime(merged_df['TIMESTAMP'].iloc[last_transition])
    filtered_df = merged_df[(merged_df['TIMESTAMP'] >= start_time) & (merged_df['TIMESTAMP'] <= end_time)]
    
    table_data = {
        'Description': [
            'Heur début', 'Heur fin', 'Efficacité', 
            'Harvested moyenne', 'Solar moyenne', 
            'Solar au debut', 'Air T au debut', 
            'Solar au fin', 'Air T au fin', 'Storage T au fin', 'Storage T au debut'
        ],
        'Value': [
            merged_df['TIMESTAMP'].iloc[first_transition],
            merged_df['TIMESTAMP'].iloc[last_transition],
            filtered_df['Harvested_m2'].mean() * 100 / filtered_df['SlrFD_W_Avg'].mean(),
            filtered_df['Harvested_m2'].mean(),
            filtered_df['SlrFD_W_Avg'].mean(),
            merged_df['SlrFD_W_Avg'].iloc[first_transition],
            merged_df['AirT_C_Avg'].iloc[first_transition],
            merged_df['SlrFD_W_Avg'].iloc[last_transition],
            merged_df['AirT_C_Avg'].iloc[last_transition],
            merged_df['T_Storage_H2O_Avg'].iloc[last_transition],
            merged_df['T_Storage_H2O_Avg'].iloc[first_transition],
        ]
    }
    
    result_df = pd.DataFrame(table_data)
    st.write(result_df)
    
def visualize_correlation():
    st.header("Correlation")
    
    merged_df = st.session_state['merged_df']
    columns = merged_df.columns.tolist()
    
    # Selecionar colunas para calcular correlação
    selected_columns = st.multiselect("Select the columns you wish to investigate", columns)
    
    # Verificar se foram selecionadas colunas suficientes para calcular correlação
    if len(selected_columns) < 2:
        st.warning("Please, select at least two columns to compare")
        return
    
    start_date = st.date_input("Select start date:", value=datetime.now().date(), key="start_correlation_date")
    start_time = st.time_input("Select start time:", value=datetime.strptime("00:00", "%H:%M").time(), key="start_correlation_time")
    end_date = st.date_input("Select end date:", value=datetime.now().date(), key="end_correlation_date")
    end_time = st.time_input("Select end time:", value=datetime.strptime("23:59", "%H:%M").time(), key="end_correlation_time")

    start_datetime = datetime.combine(start_date, start_time)
    end_datetime = datetime.combine(end_date, end_time)

    filtered_df = filter_data_by_datetime(merged_df, start_datetime, end_datetime)
    
    correlation_matrix = filtered_df[selected_columns].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title="Correlation for the whole day/days selected",
        xaxis=dict(title='Colunas'),
        yaxis=dict(title='Colunas')
    )
    
    st.plotly_chart(fig)
    
    filteredsun_df = filtered_df[filtered_df['SlrFD_W_Avg'] > 0]
    
    correlation_matrix = filteredsun_df[selected_columns].corr()
    
    fig2 = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='Viridis'
    ))
    
    fig2.update_layout(
        title="Correlation just when there is sunlight",
        xaxis=dict(title='Colunas'),
        yaxis=dict(title='Colunas')
    )
    
    st.plotly_chart(fig2)

if __name__ == "__main__":
    main()
